import streamlit as st
from data_loader import load_data, build_documents
# from models import get_embedder, get_generator
from retrieval import build_faiss_index, retrieve
from rag import generate_answer

# Domain knowledge to enrich recall documents.
DOMAIN_KNOWLEDGE = (
    "Vehicle recalls are issued when a safety-related defect is identified in a vehicle or its equipment. "
    "These recalls are often mandated by government agencies (e.g., NHTSA) or the manufacturer to prevent "
    "potential injuries or accidents. Reviewing recall history and related links helps consumers make informed decisions."
)

def main():
    # Configure the Streamlit page.
    st.set_page_config(page_title="Vehicle Recall Chatbot", layout="wide")
    st.title("Vehicle Recall Chatbot")
    st.write("""
    **Welcome!** This chatbot uses a Retrieval-Augmented Generation (RAG) pipeline backed by FAISS 
    to answer vehicle recall-related queries. The system retrieves the most relevant recall documents 
    and generates an answer using domain knowledge and language generation.
    """)
    
    # Sidebar with instructions.
    st.sidebar.header("Instructions")
    st.sidebar.write("""
    - Enter your vehicle recall query in the chat interface.
    - The system will retrieve the most relevant recall data and generate an answer.
    - The recall documents are enriched with additional safety domain information.
    """)
    
    # STEP 1: Load the recall data.
    st.info("Loading vehicle recall data...")
    df = load_data()
    st.success("Data loaded successfully!")
    
    # STEP 2: Build the document memory from the CSV data.
    # documents = build_documents(df, DOMAIN_KNOWLEDGE)

    documents = build_documents(df, DOMAIN_KNOWLEDGE)
    st.write("📝 Number of documents:", len(documents))
    st.code(documents[0][:300])  # preview first document

    
    # STEP 3: Create the FAISS index (this step is cached so it is only built once).
    with st.spinner("Building FAISS index..."):
        # embedder = get_embedder()
        # faiss_index, _ = build_faiss_index(documents, embedder)
        faiss_index, _ = build_faiss_index(documents)  
    st.success("FAISS index is ready!")
    
    # Initialize the generation model.
    generator = get_generator()
    
    # Initialize the chat history.
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.subheader("Chatbot")
    
    # Display conversation history.
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")
    
    # Chat input field.
    user_input = st.text_input("Type your question here and press Enter:")
    
    if user_input:
        # Append the user query to the conversation history.
        st.session_state.chat_history.append(("user", user_input))
        
        # STEP 4: Retrieve relevant recall documents for the query.
        with st.spinner("Retrieving relevant data..."):
            top_docs = retrieve(user_input, embedder, faiss_index, documents, k=3)
            
        # STEP 5: Generate an answer using the RAG pipeline.
        with st.spinner("Generating answer..."):
            answer = generate_answer(user_input, top_docs, generator)
        
        st.session_state.chat_history.append(("bot", answer))
        
        # Refresh the page to display the updated conversation.
        st.rerun()

if __name__ == "__main__":
    main()
