# import faiss
# import streamlit as st

# @st.cache_resource(show_spinner="Building FAISS index...")
# def build_faiss_index(documents: list, _embedder) -> (faiss.IndexFlatL2, any):
#     """
#     Build and return a FAISS index from the provided list of documents.
#     The '_embedder' is prefixed with an underscore to avoid Streamlit hashing errors.
#     """
#     st.text("Computing document embeddings...")
#     embeddings = _embedder.encode(documents, convert_to_numpy=True, show_progress_bar=True)
#     dim = embeddings.shape[1]
#     st.text(f"Embedding dimension: {dim}")
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     return index, embeddings

# def retrieve(query: str, embedder, faiss_index, documents: list, k: int = 3) -> list:
#     """
#     Retrieve the top 'k' documents for the provided query using the FAISS index.
    
#     The query is embedded using the same embedder, and the nearest neighbors (by L2 distance)
#     are returned.
#     """
#     query_embedding = embedder.encode([query], convert_to_numpy=True)
#     distances, indices = faiss_index.search(query_embedding, k)
#     retrieved_docs = [documents[idx] for idx in indices[0]]
#     return retrieved_docs

import faiss
import numpy as np
import streamlit as st
from models import compute_embeddings

@st.cache_resource(show_spinner="Building FAISS index...")
def build_faiss_index(documents: list) -> (faiss.IndexFlatL2, np.ndarray):
    embeddings = compute_embeddings(documents)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype='float32'))
    return index, embeddings

def retrieve(query: str, documents: list, index, k: int = 3) -> list:
    query_embedding = compute_embeddings([query])[0]
    distances, indices = index.search(np.array([query_embedding], dtype='float32'), k)
    return [documents[i] for i in indices[0]]
