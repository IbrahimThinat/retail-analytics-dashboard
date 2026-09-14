from pathlib import Path
import sqlite3
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from anthropic import Anthropic
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env
api_key = os.getenv("ANTHROPIC_API_key")

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="Retail Analytics & AI Assistant",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE = DATA_DIR / "retail_warehouse.db"
MODEL_FILE = MODELS_DIR / "transaction_classifier.pkl"


# ============================================================
# HELPER FUNCTIONS
# ============================================================
@st.cache_data
def load_data():
    if not DATABASE.exists():
        return pd.DataFrame()
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query("SELECT * FROM clean_transactions", conn)


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)


def execute_sql(query: str):
    """Executes SQL query against retail_warehouse.db."""
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query(query, conn)


df = load_data()
model = load_model()

st.title("📊 Retail Analytics & AI Assistant")

if df.empty:
    st.error(
        f"Database missing at `{DATABASE}`. Please run `ingest_data.py` first."
    )
    st.stop()

# ============================================================
# NAVIGATION TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(
    ["📈 Executive Overview", "🤖 Revenue Predictor (ML)", "💬 AI Data Analyst"]
)

# ------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------
with tab1:
    target_col = (
        "total_spent_clean"
        if "total_spent_clean" in df.columns
        else "total_spent"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${df[target_col].sum():,.2f}")
    c2.metric("Total Orders", f"{len(df):,}")
    c3.metric("Avg Order Value", f"${df[target_col].mean():,.2f}")

    st.divider()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Revenue by Store")
        store_rev = df.groupby("store_id")[target_col].sum().reset_index()
        fig_bar = px.bar(
            store_rev,
            x="store_id",
            y=target_col,
            color="store_id",
            labels={"store_id": "Store ID", target_col: "Revenue ($)"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.subheader("Order Value Distribution")
        fig_hist = px.histogram(
            df, x=target_col, nbins=30, labels={target_col: "Order Value ($)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------------------------------------
# TAB 2: REVENUE PREDICTOR
# ------------------------------------------------------------
with tab2:
    st.subheader("Predict Transaction Total")
    if model is None:
        st.error("Model file missing. Run `train_model.py` first.")
    else:
        c1, c2, c3 = st.columns(3)
        p = c1.number_input("Item Price ($)", min_value=1.0, value=50.0)
        q = c2.number_input("Quantity", min_value=1, value=2)
        d = c3.slider("Discount ($)", min_value=0.0, max_value=100.0, value=5.0)

        if st.button("Calculate Prediction", type="primary"):
            pred = model.predict(
                pd.DataFrame(
                    [[p, q, d]],
                    columns=["price_imputed", "quantity", "discount_imputed"],
                )
            )[0]
            st.success(f"### Predicted Total Spent: **${pred:.2f}**")

# ------------------------------------------------------------
# TAB 3: AI DATA ANALYST
# ------------------------------------------------------------
# ============================================================
# TAB 3: AI DATA ANALYST
# ============================================================
with tab3:
    st.subheader("Query Warehouse with Natural Language")
    st.write(
        "Ask questions about your sales data. Claude will write and run SQL queries dynamically."
    )

    api_key = st.text_input(
        "Enter Anthropic API Key:",
        type="password",
        help="Paste your sk-ant-... key here",
    )
    user_query = st.text_input(
        "Enter your question:",
        value="Which store generated the highest total revenue?",
    )

    if st.button("Run Query Agent", type="primary"):
        if not api_key:
            st.warning("⚠️ Please input an Anthropic API Key.")
        elif not user_query:
            st.warning("⚠️ Please write a question.")
        else:
            with st.spinner("AI Agent writing and executing SQL query..."):
                try:
                    # Dynamically import the helper function
                    try:
                        from src.agent_analytics import ask_data_agent
                    except ImportError:
                        from agent_analytics import ask_data_agent

                    response_text = ask_data_agent(user_query, api_key.strip())
                    st.markdown(response_text)

                except Exception as e:
                    st.error(f"Execution Error: {e}")