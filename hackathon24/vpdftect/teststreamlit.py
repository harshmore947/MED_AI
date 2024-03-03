import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set up Streamlit app
st.title("PDF Document Question Answering")

# File uploader for PDF document
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# Function to process uploaded PDF and perform question answering
def process_pdf_and_answer_question(pdf_file, query):
    # Read text from PDF
    raw_text = ''
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    
    # Split text into smaller chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=800, chunk_overlap=200, length_function=len)
    texts = text_splitter.split_text(raw_text)
    
    # Download embeddings from OpenAI
    # Initialize OpenAIEmbeddings with API key
    embeddings = OpenAIEmbeddings(openai_api_key="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")

    # Create FAISS index from text chunks embeddings
    document_search = FAISS.from_texts(texts, embeddings)
    
    # Load question answering model
    # Initialize OpenAI with API key
    openai_instance = OpenAI(openai_api_key="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")
    chain = load_qa_chain(openai_instance, chain_type="stuff")
    
    # Perform similarity search for the query
    docs = document_search.similarity_search(query)
    
    # Run question answering model on the retrieved documents
    response = chain.run(input_documents=docs, question=query)
    
    return response

# Main part of the app
if uploaded_file is not None:
    # Display uploaded PDF
    st.subheader("Uploaded PDF:")
    st.write(uploaded_file)

    # Text input for user's query
    query = st.text_input("Enter your question:")

    # Button to perform question answering
    if st.button("Ask"):
        # Perform question answering and display the response
        response = process_pdf_and_answer_question(uploaded_file, query)
        st.subheader("Response:")
        st.write(response)
