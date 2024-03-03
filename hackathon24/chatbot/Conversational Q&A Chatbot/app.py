import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import speech_recognition as sr

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

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

def main():
    initialize_session()
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

if __name__ == "__main__":
    main()
