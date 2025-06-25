import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from openai import AzureOpenAI
import onnxruntime as rt

# ───────────────────────────────────────
# 0) AZURE OPENAI CONFIGURATION
# ───────────────────────────────────────
AZURE_ENDPOINT       = "https://group7project.openai.azure.com/"
AZURE_DEPLOYMENT     = "gpt-4o-mini"
AZURE_OPENAI_API_KEY = "7CqvJEXBe6eFMK18yVr9jB811IyfIGbw2FqxCZREkMmqwJWQNj4JJQQJ99BEACYeBjFXJ3w3AAAAACOGSx58"

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-05-01-preview",
)

# ───────────────────────────────────────
# 1) LOAD & CACHE MODEL ARTIFACTS
# ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_artifacts():
    # classical model
    xgb_model     = joblib.load("xgb_full_model.joblib")
    # ONNX deep model session + label encoder
    onnx_sess     = rt.InferenceSession("investor_model.onnx", providers=["CPUExecutionProvider"])
    le            = joblib.load("investor_label_encoder.joblib")
    # feature metadata
    FEATURE_COLS  = joblib.load("feature_columns.joblib")
    CAT_COLS      = joblib.load("categorical_columns.joblib")
    TFIDF_VECTORS = joblib.load("tfidf_vectorizers.joblib")
    # data tables
    PORTFOLIO     = pd.read_csv("portfolio_companies.csv")
    PE_FUNDS      = pd.read_csv("pe_funds.csv")
    return xgb_model, onnx_sess, le, FEATURE_COLS, CAT_COLS, TFIDF_VECTORS, PORTFOLIO, PE_FUNDS

xgb_model, onnx_sess, le, FEATURE_COLS, CAT_COLS, TFIDF_VECTORS, PORTFOLIO, PE_FUNDS = load_artifacts()

# ───────────────────────────────────────
# 2) NLTK SETUP & TEXT CLEANER
# ───────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    txt = str(text).lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    tokens = [w for w in txt.split() if w.isalpha() and w not in stop_words]
    return " ".join(lemmatizer.lemmatize(w, wordnet.NOUN) for w in tokens if len(w) > 1)

# ───────────────────────────────────────
# 3) UI: MODEL SELECTOR + COMPANY SELECTOR
# ───────────────────────────────────────
st.title("PE-Investor Recommender + Azure Insights")
model_choice = st.radio("Choose recommendation model:", ("XGBoost", "Deep Learning"))
company      = st.selectbox("Pick a portfolio company:", PORTFOLIO["Target"].unique())

# ───────────────────────────────────────
# 4) BUILD CANDIDATES & METADATA
# ───────────────────────────────────────
comp_row = PORTFOLIO[PORTFOLIO["Target"] == company].iloc[[0]]
clean_sector    = clean_text(comp_row["Sector"].iloc[0] or "")
clean_subsector = clean_text(comp_row["Subsector"].iloc[0] or "")

cands = PE_FUNDS.copy().rename(columns={"PE_Name":"investor_id"})
cands["Target"] = company

# merge portfolio metadata
cands = cands.merge(
    comp_row[["Target","Target HQ","PE HQ","source_country_tab"]],
    on="Target", how="left"
)

# merge fund metadata
pe_meta = PE_FUNDS[[
    "PE_Name","source_country_tab","Office in Spain (Y/N)",
    "Top Geographies","Sectors"
]].rename(columns={
    "PE_Name":            "investor_id",
    "source_country_tab": "source_country_tab_PE_fund",
    "Top Geographies":    "Fund_Top_Geographies",
    "Sectors":            "Fund_Sectors"
})
cands = cands.merge(pe_meta, on="investor_id", how="left")

# ───────────────────────────────────────
# 5) BUILD NLP FIELDS
# ───────────────────────────────────────
cands["NLP_Sector"]         = clean_sector
cands["NLP_Subsector"]      = clean_subsector
cands["NLP_Sectors"]        = cands["Fund_Sectors"].fillna("").apply(clean_text)
cands["NLP_Top Geographies"]= cands["Fund_Top_Geographies"].fillna("").apply(clean_text)

# ───────────────────────────────────────
# 6) VECTORIZE & ASSEMBLE FEATURES
# ───────────────────────────────────────
for col, vect in TFIDF_VECTORS.items():
    mat = vect.transform(cands[f"NLP_{col}"]).toarray()
    df_tfidf = pd.DataFrame(
        mat,
        columns=[f"TFIDF_{col}_{i}" for i in range(mat.shape[1])],
        index=cands.index
    )
    cands = pd.concat([cands, df_tfidf], axis=1)

valid_cat = [c for c in CAT_COLS if c in cands.columns]
categ     = pd.get_dummies(cands[valid_cat], drop_first=True)

X_new = (
    pd.concat([cands.filter(regex="^TFIDF_"), categ], axis=1)
      .reindex(columns=FEATURE_COLS, fill_value=0)
)

# ───────────────────────────────────────
# 7) PREDICT & ATTACH SCORES
# ───────────────────────────────────────
# XGBoost binary probability
probs_xgb = xgb_model.predict_proba(X_new)[:,1]
cands["score_xgb"]  = probs_xgb

# ONNX deep model inference
inp_name  = onnx_sess.get_inputs()[0].name
onnx_outs = onnx_sess.run(None, {inp_name: X_new.to_numpy().astype(np.float32)})
# assume the second tensor is class probabilities: shape (n_candidates, n_classes)
deep_preds = onnx_outs[1]
class_idx  = le.transform(cands["investor_id"])
probs_deep = deep_preds[np.arange(len(cands)), class_idx]
cands["score_deep"] = probs_deep

# pick top-10 based on user’s choice
score_col = "score_xgb" if model_choice=="XGBoost" else "score_deep"
cands_sorted = cands.nlargest(10, score_col)[["investor_id", score_col]]

st.subheader(f"Top 10 Investors ({model_choice}) for {company}")
st.table(
    cands_sorted.rename(columns={score_col:"score"})
                .style.format({"score":"{:.2%}"})
)

# ───────────────────────────────────────
# 8) AUTO-GENERATED INSIGHTS via Azure
# ───────────────────────────────────────
items = "\n".join(
    f"{i+1}. {row.investor_id} ({row[score_col]:.1%})"
    for i, row in cands_sorted.reset_index(drop=True).iterrows()
)

system = (
    "You are a knowledgeable private-equity research assistant. "
    "You have a list of the top 3 recommended PE investors."
)
user = f"""
Here are the top 3 investors (by {model_choice}) for {company}:
{items}

1) Explain why they match (geography, sector, past deals).
2) Provide one recent news headline or URL for each demonstrating relevant interest.
"""

with st.spinner("Generating insights…"):
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {"role":"system",  "content": system},
            {"role":"user",    "content": user},
        ],
        temperature=0.7,
        max_tokens=500,
    )

insight = response.choices[0].message.content
st.markdown("## 💡 AI-Generated Insights")
st.write(insight)
