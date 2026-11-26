import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Deteksi Penipuan Kartu Kredit", page_icon="💳", layout="wide")

@st.cache_resource
def load_model():
    bundle = joblib.load("model_fraud_rf.joblib")
    return bundle["model"], bundle["columns"]

model, columns = load_model()

st.title("💳 Deteksi Transaksi Kartu Kredit Penipuan")
st.caption("Model: Random Forest | Dataset: Credit Card Fraud Detection (ULB, Kaggle)")

threshold = st.sidebar.slider("Threshold penipuan", 0.05, 0.95, 0.50, 0.05,
    help="Makin rendah, makin banyak transaksi ditandai (recall naik, precision turun).")
st.sidebar.info("Fitur V1-V28 adalah hasil PCA (data dianonimkan), jadi input dilakukan lewat file CSV.")

uploaded = st.file_uploader("Unggah CSV transaksi (kolom: Time, V1-V28, Amount)", type="csv")
use_sample = st.button("Gunakan contoh data")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv("sample_transaksi.csv")

if df is not None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        st.error(f"Kolom tidak lengkap: {missing}")
    else:
        proba = model.predict_proba(df[columns])[:, 1]
        out = df.copy()
        out["Probabilitas_Penipuan"] = proba.round(4)
        out["Prediksi"] = (proba >= threshold).map({True: "PENIPUAN", False: "Normal"})
        c1, c2, c3 = st.columns(3)
        c1.metric("Total transaksi", len(out))
        c2.metric("Ditandai penipuan", int((out["Prediksi"] == "PENIPUAN").sum()))
        c3.metric("Probabilitas tertinggi", f"{proba.max():.2%}")
        st.dataframe(
            out[["Time", "Amount", "Probabilitas_Penipuan", "Prediksi"]]
            .sort_values("Probabilitas_Penipuan", ascending=False),
            use_container_width=True)
        st.download_button("Unduh hasil (CSV)", out.to_csv(index=False), "hasil_prediksi.csv")
else:
    st.info("Unggah CSV atau klik 'Gunakan contoh data'.")
