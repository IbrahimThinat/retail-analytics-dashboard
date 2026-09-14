import os
from pathlib import Path
import sqlite3
from anthropic import Anthropic
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from .env file
load_dotenv()

BASE_DIR = (
    Path(__file__).resolve().parent.parent
    if (Path(__file__).resolve().parent / "data").exists() is False
    else Path(__file__).resolve().parent
)
DATABASE = BASE_DIR / "data" / "retail_warehouse.db"


def execute_sql_query(query: str) -> pd.DataFrame:
    """Executes a SQL query against the retail SQLite warehouse."""
    with sqlite3.connect(DATABASE) as conn:
        return pd.read_sql_query(query, conn)


def ask_data_agent(question: str, api_key: str = None) -> str:
    """Uses Anthropic Claude to turn natural language into SQL and summarize results."""
    # Priority: passed API key > .env variable
    resolved_api_key = (
        api_key if api_key and api_key.strip() else os.getenv("ANTHROPIC_API_KEY")
    )

    if not resolved_api_key:
        return "⚠️ Error: No Anthropic API Key provided in Streamlit or .env file."

    client = Anthropic(api_key=resolved_api_key.strip())

    system_prompt = """
    You are an expert AI Data Analyst for a retail company.
    You have access to an SQLite database with the table `clean_transactions`.

    Table Schema for `clean_transactions`:
    - store_id (INTEGER/TEXT): Store identifier (e.g., 101, 102, 103, 104)
    - price_imputed (FLOAT): Item price
    - quantity (INTEGER): Units sold
    - discount_imputed (FLOAT): Discount applied
    - total_spent_clean (FLOAT): Total order revenue

    When asked a question:
    1. Write a valid SQLite query.
    2. Enclose the query inside a SQL code block like this:
       ```sql
       SELECT store_id, SUM(total_spent_clean) AS revenue FROM clean_transactions GROUP BY store_id;
       ```
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )

    content = response.content[0].text

    # Extract SQL query from response if present
    if "```sql" in content:
        sql_query = content.split("```sql")[1].split("```")[0].strip()
        try:
            df_result = execute_sql_query(sql_query)

            # Format answer with query results
            summary_prompt = f"""
            User Question: {question}
            SQL Executed: {sql_query}
            Query Result Table:
            {df_result.to_string(index=False)}

            Please summarize the answer clearly for a business stakeholder.
            """

            final_response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": summary_prompt}],
            )

            return (
                final_response.content[0].text
                + f"\n\n**Executed SQL:**\n```sql\n{sql_query}\n```"
            )
        except Exception as e:
            return f"Error executing generated SQL: {e}\n\nDrafted SQL:\n```sql\n{sql_query}\n```"

    return content