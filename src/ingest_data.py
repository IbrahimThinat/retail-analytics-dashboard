from pathlib import Path
import sqlite3
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

RAW_CSV = DATA_DIR / "raw_transactions.csv"
CLEAN_CSV = DATA_DIR / "clean_transactions.csv"
DATABASE = DATA_DIR / "retail_warehouse.db"


# ============================================================
# LOAD DATA
# ============================================================

def load_raw_data():
    """Load the raw transactions CSV."""

    if not RAW_CSV.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_CSV}"
        )

    df = pd.read_csv(RAW_CSV)

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):
    """Clean and prepare transaction data."""

    df = df.copy()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    # Remove completely duplicated rows
    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "price",
        "quantity",
        "discount",
        "total_amount"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # --------------------------------------------------------
    # Fill missing numeric values
    # --------------------------------------------------------

    for column in numeric_columns:
        if column in df.columns and df[column].isna().any():
            df[column] = df[column].fillna(
                df[column].median()
            )

    # --------------------------------------------------------
    # Fill missing categorical values
    # --------------------------------------------------------

    categorical_columns = [
        "store",
        "product",
        "category",
        "payment_method"
    ]

    for column in categorical_columns:
        if column in df.columns and df[column].isna().any():

            mode = df[column].mode()

            if not mode.empty:
                df[column] = df[column].fillna(mode.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

    # --------------------------------------------------------
    # Clean text columns
    # --------------------------------------------------------

    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype(str).str.strip()

    # --------------------------------------------------------
    # Make sure quantity and price are valid
    # --------------------------------------------------------

    if "quantity" in df.columns:
        df = df[df["quantity"] >= 0]

    if "price" in df.columns:
        df = df[df["price"] >= 0]

    # --------------------------------------------------------
    # Create total_amount if it does not exist
    # --------------------------------------------------------

    if (
        "total_amount" not in df.columns
        and "price" in df.columns
        and "quantity" in df.columns
    ):
        df["total_amount"] = (
            df["price"] *
            df["quantity"]
        )

    return df


# ============================================================
# SAVE CLEAN CSV
# ============================================================

def save_clean_data(df):
    """Save cleaned transactions to CSV."""

    df.to_csv(
        CLEAN_CSV,
        index=False
    )


# ============================================================
# LOAD DATA INTO SQLITE
# ============================================================

def load_into_database(raw_df, clean_df):
    """Create SQLite database tables."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with sqlite3.connect(DATABASE) as conn:

        # Raw data table
        raw_df.to_sql(
            "raw_transactions",
            conn,
            if_exists="replace",
            index=False
        )

        # Clean data table
        clean_df.to_sql(
            "clean_transactions",
            conn,
            if_exists="replace",
            index=False
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("RETAIL DATA INGESTION PIPELINE")
    print("=" * 60)

    # 1. Load raw data
    print("\n[1/5] Loading raw data...")

    raw_df = load_raw_data()

    print(
        f"Loaded {len(raw_df):,} raw rows."
    )

    # 2. Clean data
    print("\n[2/5] Cleaning data...")

    clean_df = clean_data(raw_df)

    print(
        f"Clean dataset contains "
        f"{len(clean_df):,} rows."
    )

    # 3. Save clean CSV
    print("\n[3/5] Saving clean CSV...")

    save_clean_data(clean_df)

    print(
        f"Saved: {CLEAN_CSV}"
    )

    # 4. Create SQLite database
    print("\n[4/5] Creating SQLite database...")

    load_into_database(
        raw_df,
        clean_df

    )

    print(
        f"Database created: {DATABASE}"
    )

    # 5. Verify database
    print("\n[5/5] Verifying database...")

    with sqlite3.connect(DATABASE) as conn:

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name;
            """,
            conn
        )

    print("\nDatabase tables:")

    for table in tables["name"]:
        print(f"  ✓ {table}")

    print("\n" + "=" * 60)
    print("INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()