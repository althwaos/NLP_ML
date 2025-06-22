import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# ───────────────────────────────
# 1) LOAD & CACHE ARTIFACTS
# ───────────────────────────────
@st.cache_resource
def load_artifacts():
    model             = joblib.load("xgb_full_model.joblib")
    FEATURE_COLS      = joblib.load("feature_columns.joblib")
    CAT_COLS          = joblib.load("categorical_columns.joblib")
    TFIDF_VECTORS     = joblib.load("tfidf_vectorizers.joblib")
    portfolio         = pd.read_csv("portfolio_companies.csv")
    pe_funds          = pd.read_csv("pe_funds.csv")
    return model, FEATURE_COLS, CAT_COLS, TFIDF_VECTORS, portfolio, pe_funds

model, FEATURE_COLS, CAT_COLS, TFIDF_VECTORS, PORTFOLIO, PE_FUNDS = load_artifacts()
# Ensure NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# text cleaner (same as training)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def clean_text(text):
    txt = str(text).lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    tokens = [w for w in txt.split() if w.isalpha() and w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w, wordnet.NOUN) for w in tokens if len(w)>1)

# ───────────────────────────────
# 2) UI: select a company
# ───────────────────────────────
st.title("PE-Investor Recommender")
company = st.selectbox("Pick a portfolio company:", PORTFOLIO["Target"].unique())

# ───────────────────────────────
# 3) BUILD CANDIDATES
# ───────────────────────────────
comp_row = PORTFOLIO[PORTFOLIO["Target"] == company].iloc[[0]]
# Pre-clean these two for the company itself (we'll broadcast them)
for col in ["Sector","Subsector"]:
    comp_row[f"NLP_{col}"] = comp_row[col].apply(clean_text)

cands = PE_FUNDS.copy()
cands["Target"] = company
cands = cands.rename(columns={"PE_Name": "investor_id"})

# ───────────────────────────────
# 4) MERGE METADATA & NLP
# ───────────────────────────────
# Include Sector & Subsector in this merge so those columns exist on cands
cands = cands.merge(
    comp_row[[
        "Target",
        "Target HQ",
        "PE HQ",
        "source_country_tab",
        "Sector",       # <— added
        "Subsector"     # <— added
    ]],
    on="Target", how="left"
)

cands = cands.merge(
    PE_FUNDS[[
        "PE_Name",
        "source_country_tab",
        "Office in Spain (Y/N)",
        "Top Geographies",
        "Sectors"
    ]].rename(columns={
        "PE_Name": "investor_id",
        "source_country_tab": "source_country_tab_PE_fund"
    }),
    on="investor_id", how="left"
)

# Now we can clean all four text columns without KeyErrors:
for col in ["Sector","Subsector","Sectors","Top Geographies"]:
    cands[f"NLP_{col}"] = cands[col].fillna("").apply(clean_text)

# ───────────────────────────────
# 5) VECTORIZE
# ───────────────────────────────
for col, vect in TFIDF_VECTORS.items():
    key = f"NLP_{col}"
    mat = vect.transform(cands[key]).toarray()
    tfdf = pd.DataFrame(
        mat,
        columns=[f"TFIDF_{col}_{i}" for i in range(mat.shape[1])],
        index=cands.index
    )
    cands = pd.concat([cands, tfdf], axis=1)

categ = pd.get_dummies(cands[CAT_COLS], drop_first=True)
X_new = pd.concat([
    cands.filter(regex="^TFIDF_"),
    categ
], axis=1).reindex(columns=FEATURE_COLS, fill_value=0)

# ───────────────────────────────
# 6) PREDICT & DISPLAY TOP-10
# ───────────────────────────────
probs = model.predict_proba(X_new)[:,1]
cands["score"] = probs
top10 = cands.nlargest(10, "score")[["investor_id","score"]]

st.subheader(f"Top 10 Investors for {company}")
st.table(top10.style.format({"score":"{:.2%}"}))