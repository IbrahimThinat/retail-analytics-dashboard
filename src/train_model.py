from pathlib import Path
import sqlite3
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================
# PROJECT PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATABASE = DATA_DIR / "retail_warehouse.db"
MODEL_FILE = MODELS_DIR / "transaction_classifier.pkl"


def load_clean_data():
    """Load clean data from SQLite database."""
    if not DATABASE.exists():
        raise FileNotFoundError(
            f"Database file not found at: {DATABASE}. Run ingest_data.py first."
        )

    with sqlite3.connect(DATABASE) as conn:
        df = pd.read_sql_query("SELECT * FROM clean_transactions", conn)
    return df


def train_and_evaluate():
    """Train Random Forest model to predict order revenue."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_data()

    # Identify matching columns dynamically
    price_col = "price_imputed" if "price_imputed" in df.columns else "price"
    discount_col = (
        "discount_imputed" if "discount_imputed" in df.columns else "discount"
    )
    target_col = (
        "total_spent_clean"
        if "total_spent_clean" in df.columns
        else "total_spent" if "total_spent" in df.columns else "total_amount"
    )

    features = [price_col, "quantity", discount_col]
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate Model
    predictions = model.predict(X_test)
    try:
        # For newer scikit-learn versions
        from sklearn.metrics import root_mean_squared_error

        rmse = root_mean_squared_error(y_test, predictions)
    except ImportError:
        # Fallback for standard scikit-learn versions
        rmse = mean_squared_error(y_test, predictions, squared=False)

    r2 = r2_score(y_test, predictions)

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    print(f"Features used : {features}")
    print(f"Target column : {target_col}")
    print(f"RMSE          : {rmse:.4f}")
    print(f"R² Score      : {r2:.4f}")

    # Save Artifact
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved successfully to: {MODEL_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()