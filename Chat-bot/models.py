# from sentence_transformers import SentenceTransformer
# from transformers import pipeline, set_seed
# import streamlit as st

# # Model names
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# GENERATION_MODEL_NAME = "gpt2"

# # Set seed for reproducibility
# set_seed(42)

# @st.cache_resource(show_spinner=False)
# def get_embedder() -> SentenceTransformer:
#     """
#     Loads and returns the sentence transformer model for generating embeddings.
#     The result is cached for faster re-use.
#     """
#     return SentenceTransformer(EMBEDDING_MODEL_NAME)

# @st.cache_resource(show_spinner=False)
# def get_generator():
#     """
#     Loads and returns the text-generation pipeline (e.g., GPT-2) for producing answers.
#     """
#     return pipeline("text-generation", model=GENERATION_MODEL_NAME)


import openai
import streamlit as st
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd

load_dotenv('keys.env')

# key = os.getenv("openai")


openai.api_key = os.getenv("openai")

MAX_DOC_LENGTH = 3000  # safe for token limit of ada-002

def truncate_text(text: str, max_length: int = MAX_DOC_LENGTH) -> str:
    if not isinstance(text, str):
        return ""
    return text[:max_length] + "..." if len(text) > max_length else text

def clean_documents(texts: list) -> list:
    cleaned = []
    for i, text in enumerate(texts):
        if pd.isna(text):
            continue
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue
        text = truncate_text(text)
        if text:
            cleaned.append(text)
    return cleaned

def compute_embeddings(texts: list) -> list:
    cleaned_texts = clean_documents(texts)

    if not cleaned_texts:
        st.error("🛑 No valid texts to embed.")
        raise ValueError("No valid texts to embed.")

    # Debugging: Print the first one you're trying to send
    st.write("🧪 Sending texts to OpenAI Embedding API. First sample:")
    st.code(cleaned_texts[0][:300])

    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002", # Try new text emedding models
            input=cleaned_texts
        )
        return [item["embedding"] for item in response["data"]]

    except openai.error.InvalidRequestError as e:
        st.error("❌ OpenAI embedding failed: Invalid request.")
        st.exception(e)
        raise

    except Exception as e:
        st.error("❌ Unexpected error during embedding.")
        st.exception(e)
        raise

def generate_chat_completion(prompt: str, max_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant for vehicle safety and recall queries."},
        {"role": "user", "content": prompt}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.6
    )
    return response.choices[0].message.content.strip()
