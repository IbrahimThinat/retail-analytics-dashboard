import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Retail ML & Analytics Dashboard", layout="wide")


# 1. Load Data
@st.cache_data
def load_data():
    conn = sqlite3.connect("data/retail_warehouse.db")
    df = pd.read_sql_query("SELECT * FROM clean_transactions", conn)
    conn.close()
    return df


df = load_data()


# 2. Load ML Artifacts
@st.cache_resource
def load_ml_models():
    model_path = Path("models/transaction_classifier.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None


clf_model = load_ml_models()

st.title("🛒 Retail Analytics & Machine Learning Dashboard")

# Sidebar: Interactive ML Simulator
st.sidebar.header("🤖 Order Value Classifier (ML)")
sim_price = st.sidebar.number_input("Item Price ($)", min_value=1.0, value=50.0, step=5.0)
sim_qty = st.sidebar.number_input("Quantity", min_value=1, value=2, step=1)
sim_discount = st.sidebar.slider("Discount Rate", 0.0, 0.5, 0.1, step=0.05)

discount_impact = sim_price * sim_qty * sim_discount
est_total = (sim_price * sim_qty) - discount_impact

st.sidebar.markdown(f"**Estimated Order Total:** `${est_total:.2f}`")

if clf_model is not None:
    # Prepare input vector matching model features: ['price', 'quantity', 'discount', 'discount_impact']
    input_data = pd.DataFrame([[sim_price, sim_qty, sim_discount, discount_impact]],
                              columns=['price', 'quantity', 'discount', 'discount_impact'])

    prediction = clf_model.predict(input_data)[0]
    prob = clf_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.sidebar.success(f"🔥 **High-Value Order** ({prob * 100:.1f}% Probability)")
    else:
        st.sidebar.info(f"📦 **Standard Order** ({(1 - prob) * 100:.1f}% Probability)")
else:
    st.sidebar.warning("ML Model not found. Run `02_week2_ml_foundations.ipynb` to generate the model artifact.")

# Dashboard KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${df['total_spent_clean'].sum():,.2f}")
col2.metric("Total Transactions", f"{len(df):,}")
col3.metric("Avg Order Value", f"${df['total_spent_clean'].mean():,.2f}")

st.markdown("---")

# Visualizations
c1, c2 = st.columns(2)

with c1:
    st.subheader("Revenue by Store Location")
    store_rev = df.groupby("store_id")["total_spent_clean"].sum().reset_index()
    fig_bar = px.bar(store_rev, x="store_id", y="total_spent_clean",
                     color="total_spent_clean", labels={"total_spent_clean": "Revenue ($)"})
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    st.subheader("Transaction Value Distribution")
    fig_hist = px.histogram(df, x="total_spent_clean", nbins=30,
                            labels={"total_spent_clean": "Spent Amount ($)"})
    st.plotly_chart(fig_hist, use_container_width=True)