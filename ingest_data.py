import pandas as pd
import sqlite3

# Load raw CSV
df = pd.read_csv("data/raw_transactions.csv")

# Clean column headers
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Load into SQLite
conn = sqlite3.connect("data/retail_warehouse.db")
df.to_sql("raw_transactions", conn, if_exists="replace", index=False)
conn.close()