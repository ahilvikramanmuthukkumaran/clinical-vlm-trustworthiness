import streamlit as st
from validation import validate_question
from PIL import Image

st.set_page_config(page_title = "CLINICAL VLM TRUSTWORTHINESS")

st.title("Clinical Vision Language Model Trustworthiness")
st.write("Research prototype for studying the hallucinations in medical VLMs.")

image =  st.file_uploader(
    "Upload a valid medical image",
    type=["png", "jpg", "jpeg"]
)
if image is not None:
    try:
        img = Image.open(image)
        img.verify()
        
        img.seek(0)
        img = Image.open(image)
        st.image(img, caption="Uploaded medical image", use_container_width=True)
        st.success("Image file is valid.")
    except Exception:
        st.error("Invalid image file. Please upload a valid PNG or JPEG.")

question = st.text_area("Enter your clinical question:")

if st.button("Validate inputs"):
    if image is None:
        st.error("Please upload an image.")
    else:
        valid, message = validate_question(question)
        if not valid:
            st.error(message)
        else:
            st.success("Inputs received! Model integration is not connected yet.")
