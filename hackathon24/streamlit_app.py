import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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

# Function to get response from GenerativeModel
def get_gemini_response(input_text):
    model = genai.GenerativeModel("gemini-pro")
    if input_text.strip() != "":
        response = model.generate_content(input_text)
        return response.text
    else:
        return "Input text is empty. Please provide a valid input."

# Main function to run the app
def main():
    st.sidebar.title("Options")
    app_mode = st.sidebar.radio("Choose an option", ["Chatbot", "PDF Reader", "Image Reader"])

    if app_mode == "Chatbot":
        st.title("Sarcastic ChatBot")
        initialize_session()
        chatbot_interaction()

    elif app_mode == "PDF Reader":
        st.title("MED-AI Report Reader")
        pdf_interaction()

    elif app_mode == "Image Reader":
        st.title("MED-AI Image Assistance")
        image_interaction()

# Function to initialize chatbot session
def initialize_session():
    if 'flowmessages' not in st.session_state:
        st.session_state['flowmessages'] = [SystemMessage(content="You are a comedian AI assistant")]

def get_chatmodel_response(question):
    chat = ChatOpenAI(openai_api_key="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", temperature=0.5)

    # chat = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0.5)
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer = chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content


# Function to handle chatbot interaction
def chatbot_interaction():
    option = st.radio("Choose input method:", ("Text", "Voice"))
    if option == "Text":
        input_text = st.text_input("Input: ", key="input")
    else:
        st.info("Click the 'Start Recording' button and speak")
        if st.button("Start Recording"):
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                st.write("Recording...")
                audio = recognizer.listen(source)
            try:
                input_text = recognizer.recognize_google(audio)
                st.success("Recording complete!")
                st.text_input("Input: ", value=input_text, key="input")
            except sr.UnknownValueError:
                st.error("Unable to recognize speech")
            except sr.RequestError as e:
                st.error(f"Error: {e}")

    if st.button("Ask the question"):
        response = get_chatmodel_response(input_text)
        st.subheader("The Response is")
        st.write(response)

# Function to handle PDF interaction
def pdf_interaction():
    # File uploader for PDF document
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

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

# Function to handle image interaction
def image_interaction():
    # Text input for user prompt
    input_text = st.text_area("Input Prompt:")

    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    # Define your Google API key here
    GOOGLE_API_KEY = "AIzaSyDe5QKtR9Z6Yczj6U1tvpWecJV0ktBbMg4"

    # Configure GenerativeAI with the API key
    genai.configure(api_key=GOOGLE_API_KEY)

    # Initialize GenerativeModel
    model = genai.GenerativeModel("gemini-pro")


    # Button to describe the image
    submit = st.button("Describe the Image")

    # If button is clicked
    if submit:
        if input_text.strip() != "":
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

                # Generate response based on input text
                response = get_gemini_response(input_text)
                st.subheader("The Response is:")
                st.write(response)
            else:
                st.error("Please upload an image.")
        else:
            st.error("Please provide a non-empty input text.")

if __name__ == "__main__":
    main()
