import streamlit as st
import google.generativeai as genai
from PIL import Image

# Define your Google API key here
GOOGLE_API_KEY = "AIzaSyDe5QKtR9Z6Yczj6U1tvpWecJV0ktBbMg4"

# Configure GenerativeAI with the API key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize GenerativeModel
model = genai.GenerativeModel("gemini-pro")

# Function to get response from GenerativeModel
def get_gemini_response(input_text):
    if input_text.strip() != "":
        response = model.generate_content(input_text)
        return response.text
    else:
        return "Input text is empty. Please provide a valid input."

# Set Streamlit page configuration
st.set_page_config(page_title="MEDAI Image Assistance")

# Streamlit header
st.header("Application")

# Text input for user prompt
input_text = st.text_area("Input Prompt:")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

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
