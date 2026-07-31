import streamlit as st

st.set_page_config(page_title="Docker Demo", layout="centered")

st.title("🐳 Hello from Docker!")
st.write("This is a minimal Streamlit application designed to demonstrate that the Docker container is building and running successfully.")
st.success("Your fast Docker image is working perfectly!")
st.info("Note: The heavy ML libraries were excluded from this image to make it build quickly.")
