# 🛒 Retail Analytics & Insight Bot Dashboard

An end-to-end data pipeline, predictive analytics engine, and interactive dashboard built to process retail transactions, evaluate basket performance, and generate automated executive insights.

---

## 📌 Project Overview
This 10-week capstone project transforms raw, unstructured transactional logs into clean analytics assets. The system performs dynamic data cleaning, stores optimized tables in an SQLite relational database, predicts high-value transactions using machine learning, and presents real-time KPIs through a Streamlit web application.

---

## 🛠️ Tech Stack & Tools
* **Language & Core**: Python 3.14, `pathlib`
* **Data Processing & Storage**: `pandas`, `sqlite3`
* **Machine Learning**: `scikit-learn` (Binary Classification Engine)
* **Visualization & Web UI**: `streamlit`, `plotly`
* **Documentation**: Markdown, Executable Jupyter Notebooks

---

## 📁 Repository Structure
```text
PythonProject1/
├── data/
│   ├── clean_transactions.csv   # Post-imputation cleaned dataset
│   ├── raw_transactions.csv     # Raw transactional log data
│   └── retail_warehouse.db      # SQLite relational database warehouse
├── notebooks/
│   └── 01_week1_pandas_foundations.ipynb  # Exploratory analysis & pipeline logic
├── app.py                       # Main Streamlit web application
├── CAPSTONE_REPORT.md           # Executive capstone completion report
├── README.md                    # Project documentation
└── .gitignore                   # Version control exclusion parameters
```

---

## 🚀 Key Features & Functionality

### 1. Data Engineering & SQL Warehousing
* Cleans raw logs, imputes missing values using median substitution, and calculates net revenue parameters.
* Loads structured datasets into an SQLite database (`retail_warehouse.db`) to enable efficient querying with SQL window functions.

### 2. Machine Learning Pipeline
* Trains a supervised binary classifier (`scikit-learn`) on transaction metrics (e.g., item quantities, pricing parameters) to identify high-value customer orders (top 20th percentile).

### 3. Interactive Streamlit Dashboard (`app.py`)
* Displays key metrics including Total Revenue ($294,215.02), Order Volume (1,000 transactions), and Average Ticket Size ($294.22).
* Renders interactive Plotly visual charts analyzing sales performance across store locations and product categories.
* Features a live **Prediction Simulator** sidebar allowing users to test input parameters against the machine learning model in real time.

### 4. Executive Insight Bot Payload Engine
* Converts key retail metrics into structured JSON payloads formatted specifically for automated executive summary generation.

---

## 💻 How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/IbrahimThinat/retail-analytics-dashboard.git](https://github.com/IbrahimThinat/retail-analytics-dashboard.git)
   cd retail-analytics-dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas sqlite3 scikit-learn plotly streamlit
   ```

3. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
   *The application will launch in your web browser at `http://localhost:8501`.*
