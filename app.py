import streamlit as st

st.set_page_config(page_title = "CLINICAL VLM TRUSTWORTHINESS")

st.title("Clinical Vision Language Model Trustworthiness")
st.write("Research prototype for studying the hallucinations in medical VLMs.")

image =  st.file_uploader(
    "Upload a valid medical image",
    type=["png", "jpg", "jprg"]
)

question = st.text_area("Enter your clinical question:")

if st.button("Validate inputs"):
    if image is None:
        st.error("Please upload an image.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        st.success("Inputs received! Model integration is not connected yet.")
        