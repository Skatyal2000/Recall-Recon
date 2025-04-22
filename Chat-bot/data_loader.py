import pandas as pd
import streamlit as st

# Path to your CSV dataset (adjust this path if needed)
DATA_PATH = 'vehicle_recalls-1.csv'

@st.cache_data(show_spinner=False)
def load_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the CSV data containing vehicle recall information.
    This function caches the result so the data is loaded only once.
    """
    df = pd.read_csv(data_path)
    return df

def build_documents(df: pd.DataFrame, domain_knowledge: str) -> list:
    """
    Build document entries from each row of the dataframe.
    
    Each document includes the recall ID, vehicle model, summary, link, 
    along with additional domain-specific recall knowledge.
    """
    documents = []
    for _, row in df.iterrows():
        recall_id = str(row.get("Recall_ID", "N/A"))
        model = str(row.get("Vehicle_Model", "Unknown Model"))
        summary = str(row.get("Summary", "No summary provided."))
        link = str(row.get("Link", "No link provided."))
        doc_text = (
            f"Recall ID: {recall_id}\n"
            f"Vehicle Model: {model}\n"
            f"Summary: {summary}\n"
            f"More info: {link}\n"
            f"Domain Info: {domain_knowledge}"
        )
        documents.append(doc_text)
    return documents
