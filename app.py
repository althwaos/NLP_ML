import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import google.generativeai as genai
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# ───────────────────────────────────────
# 0) CONFIGURE GEMINI (hard-coded key)
# ───────────────────────────────────────
genai.configure(api_key="AIzaSyDy_17Hn9m6Zd3CAeOxvLdJjTlLZizdttk")

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
# 2) NLTK SETUP & CLEANER
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
st.title("PE-Investor Recommender + Insights")
company = st.selectbox("Pick a portfolio company:", PORTFOLIO["Target"].unique())

# ───────────────────────────────────────
# 4) BUILD CANDIDATES & METADATA
# ───────────────────────────────────────
comp_row = PORTFOLIO[PORTFOLIO["Target"] == company].iloc[[0]]
clean_sector    = clean_text(comp_row["Sector"].iloc[0] or "")
clean_subsector = clean_text(comp_row["Subsector"].iloc[0] or "")

cands = PE_FUNDS.copy()
cands["Target"] = company
cands = cands.rename(columns={"PE_Name": "investor_id"})

# merge portfolio metadata
cands = cands.merge(
    comp_row[["Target","Target HQ","PE HQ","source_country_tab"]],
    on="Target", how="left"
)

# merge fund metadata (renamed)
pe_meta = PE_FUNDS[[
    "PE_Name","source_country_tab","Office in Spain (Y/N)",
    "Top Geographies","Sectors"
]].rename(columns={
    "PE_Name":              "investor_id",
    "source_country_tab":   "source_country_tab_PE_fund",
    "Top Geographies":      "Fund_Top_Geographies",
    "Sectors":              "Fund_Sectors"
})
cands = cands.merge(pe_meta, on="investor_id", how="left")

# ───────────────────────────────────────
# 5) BUILD NLP FIELDS
# ───────────────────────────────────────
cands["NLP_Sector"]    = clean_sector
cands["NLP_Subsector"] = clean_subsector
cands["NLP_Sectors"]         = cands["Fund_Sectors"].fillna("").apply(clean_text)
cands["NLP_Top Geographies"] = cands["Fund_Top_Geographies"].fillna("").apply(clean_text)

# ───────────────────────────────────────
# 6) VECTORIZE & ASSEMBLE FEATURES
# ───────────────────────────────────────
for col, vect in TFIDF_VECTORS.items():
    key = f"NLP_{col}"
    mat = vect.transform(cands[key]).toarray()
    df_tfidf = pd.DataFrame(
        mat,
        columns=[f"TFIDF_{col}_{i}" for i in range(mat.shape[1])],
        index=cands.index
    )
    cands = pd.concat([cands, df_tfidf], axis=1)

valid_cat = [c for c in CAT_COLS if c in cands.columns]
categ = pd.get_dummies(cands[valid_cat], drop_first=True)

X_new = (
    pd.concat([cands.filter(regex="^TFIDF_"), categ], axis=1)
      .reindex(columns=FEATURE_COLS, fill_value=0)
)

# ───────────────────────────────────────
# 7) PREDICT & SHOW TOP-10
# ───────────────────────────────────────
probs = model.predict_proba(X_new)[:,1]
cands["score"] = probs

top10 = cands.nlargest(10, "score")[["investor_id","score"]]
st.subheader(f"Top 10 Investors for {company}")
st.table(top10.style.format({"score": "{:.2%}"}))

# ───────────────────────────────────────
# 8) AUTO-GENERATED GEMINI INSIGHTS
# ───────────────────────────────────────
items = "\n".join(
    f"{i+1}. {row.investor_id} ({row.score:.1%})"
    for i, row in top10.head(3).iterrows()
)

system = "You are a knowledgeable private-equity research assistant."
user   = f"""
I’ve recommended these top 3 investors for {company}:
{items}

1) Summarize why they match (geography, sector focus, past deals).
2) For each, list one recent public news title or URL showing their interest.
"""

response = genai.chat.completions.create(
    model="gemini-pro",
    temperature=0.7,
    messages=[
        {"author": "system", "content": system},
        {"author": "user",   "content": user},
    ],
)
insight = response.choices[0].message.content

st.markdown("## 💡 AI-Generated Insights")
st.write(insight)
