import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# ───────────────────────────────────────
# 1) LOAD & CACHE ARTIFACTS
# ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model         = joblib.load("xgb_full_model.joblib")
    FEATURE_COLS  = joblib.load("feature_columns.joblib")
    CAT_COLS      = joblib.load("categorical_columns.joblib")
    TFIDF_VECTORS = joblib.load("tfidf_vectorizers.joblib")
    PORTFOLIO     = pd.read_csv("portfolio_companies.csv")
    PE_FUNDS      = pd.read_csv("pe_funds.csv")
    return model, FEATURE_COLS, CAT_COLS, TFIDF_VECTORS, PORTFOLIO, PE_FUNDS

model, FEATURE_COLS, CAT_COLS, TFIDF_VECTORS, PORTFOLIO, PE_FUNDS = load_artifacts()

# ───────────────────────────────────────
# 2) NLTK SETUP & TEXT CLEANER
# ───────────────────────────────────────
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    txt = str(text).lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    tokens = [w for w in txt.split() if w.isalpha() and w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w, wordnet.NOUN) for w in tokens if len(w) > 1)

# ───────────────────────────────────────
# 3) UI: COMPANY SELECTOR
# ───────────────────────────────────────
st.title("PE-Investor Recommender")
company = st.selectbox("Pick a portfolio company:", PORTFOLIO["Target"].unique())

# ───────────────────────────────────────
# 4) BUILD CANDIDATES
# ───────────────────────────────────────
# a) One-row company metadata
comp_row = PORTFOLIO[PORTFOLIO["Target"] == company].iloc[[0]]

# b) Pre-clean company’s Sector/Subsector
clean_sector    = clean_text(comp_row["Sector"].iloc[0] or "")
clean_subsector = clean_text(comp_row["Subsector"].iloc[0] or "")

# c) Copy all PE funds & attach the company
cands = PE_FUNDS.copy()
cands["Target"] = company
cands = cands.rename(columns={"PE_Name": "investor_id"})

# ───────────────────────────────────────
# 5) MERGE METADATA
# ───────────────────────────────────────
# Portfolio metadata
cands = cands.merge(
    comp_row[["Target","Target HQ","PE HQ","source_country_tab"]],
    on="Target", how="left"
)

# Fund metadata — rename text cols to avoid conflict
pe_meta = PE_FUNDS[[
    "PE_Name",
    "source_country_tab",
    "Office in Spain (Y/N)",
    "Top Geographies",
    "Sectors"
]].rename(columns={
    "PE_Name": "investor_id",
    "source_country_tab": "source_country_tab_PE_fund",
    "Top Geographies": "Fund_Top_Geographies",
    "Sectors": "Fund_Sectors"
})
cands = cands.merge(pe_meta, on="investor_id", how="left")

# ───────────────────────────────────────
# 6) BUILD ALL 4 NLP_* COLUMNS
# ───────────────────────────────────────
# Company-wide (broadcast)
cands["NLP_Sector"]    = clean_sector
cands["NLP_Subsector"] = clean_subsector

# Fund-specific (row-by-row)
cands["NLP_Sectors"]         = cands["Fund_Sectors"].fillna("").apply(clean_text)
cands["NLP_Top Geographies"] = cands["Fund_Top_Geographies"].fillna("").apply(clean_text)

# ───────────────────────────────────────
# 7) VECTORIZE & ASSEMBLE FEATURES
# ───────────────────────────────────────
# a) TF-IDF per text column
for col, vect in TFIDF_VECTORS.items():
    key = f"NLP_{col}"
    mat = vect.transform(cands[key]).toarray()
    df_tfidf = pd.DataFrame(
        mat,
        columns=[f"TFIDF_{col}_{i}" for i in range(mat.shape[1])],
        index=cands.index
    )
    cands = pd.concat([cands, df_tfidf], axis=1)

# b) One-hot encode only the valid categorical cols
valid_cat_cols = [c for c in CAT_COLS if c in cands.columns]
categ = pd.get_dummies(cands[valid_cat_cols], drop_first=True)

# c) Build and align final feature matrix
X_new = pd.concat([
    cands.filter(regex="^TFIDF_"),
    categ
], axis=1).reindex(columns=FEATURE_COLS, fill_value=0)

# ───────────────────────────────────────
# 8) PREDICT & SHOW TOP-10
# ───────────────────────────────────────
probs = model.predict_proba(X_new)[:,1]
cands["score"] = probs

top10 = cands.nlargest(10, "score")[["investor_id","score"]]
st.subheader(f"Top 10 Investors for {company}")
st.table(top10.style.format({"score": "{:.2%}"}))
