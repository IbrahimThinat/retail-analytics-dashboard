import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Retail Warehouse Analytics", layout="wide")

project_root = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
db_path = project_root / "data" / "retail_warehouse.db"

@st.cache_data
def load_data():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM clean_transactions", conn)
    conn.close()
    return df

try:
    df = load_data()

    # --- SIDEBAR: Live Predictor Simulator ---
    st.sidebar.header("🎯 Transaction Classifier")
    st.sidebar.markdown("Simulate order features to check high-value status.")

    input_price = st.sidebar.number_input("Item Price ($)", min_value=1.0, max_value=1000.0, value=150.0)
    input_qty = st.sidebar.slider("Quantity", min_value=1, max_value=20, value=3)
    input_discount = st.sidebar.slider("Discount Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

    calc_total = (input_price * input_qty) * (1 - input_discount)
    high_val_cutoff = df['total_spent_clean'].quantile(0.80)

    st.sidebar.markdown(f"**Estimated Total:** ${calc_total:,.2f}")
    if calc_total > high_val_cutoff:
        st.sidebar.success(f"High-Value Order (Above ${high_val_cutoff:,.2f} threshold)")
    else:
        st.sidebar.info(f"Standard Order (Below ${high_val_cutoff:,.2f} threshold)")

    # --- MAIN PAGE ---
    st.title("Retail Analytics & Insight Dashboard")
    st.markdown("Real-time executive summaries, store performance, and transaction analytics.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${df['total_spent_clean'].sum():,.2f}")
    col2.metric("Total Transactions", len(df))
    col3.metric("Average Ticket Size", f"${df['total_spent_clean'].mean():,.2f}")

    st.divider()

    left_col, right_col = st.columns(2)

    with left_col:
        st.subheader("Revenue by Store")
        store_rev = df.groupby("store_id")["total_spent_clean"].sum().reset_index()
        fig_bar = px.bar(store_rev, x="store_id", y="total_spent_clean",
                         labels={"total_spent_clean": "Revenue ($)", "store_id": "Store ID"},
                         color="total_spent_clean", color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar, use_container_width=True)

    with right_col:
        st.subheader("Transaction Value Distribution")
        fig_hist = px.histogram(df, x="total_spent_clean", nbins=30,
                                labels={"total_spent_clean": "Total Spent ($)"},
                                color_discrete_sequence=["#1f77b4"])
        st.plotly_chart(fig_hist, use_container_width=True)

    # --- INSIGHT BOT PANEL ---
    st.divider()
    st.subheader("🤖 Insight Bot Executive Summary")
    st.info(f"Top performing location is Store {int(df.groupby('store_id')['total_spent_clean'].sum().idxmax())} "
            f"generating ${df.groupby('store_id')['total_spent_clean'].sum().max():,.2f}. "
            f"High-value transactions represent the top 20% of orders exceeding ${high_val_cutoff:,.2f}.")

except Exception as e:
    st.error(f"Error loading database file at {db_path}: {e}")
