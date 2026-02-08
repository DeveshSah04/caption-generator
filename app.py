import streamlit as st
from predict import generate_caption
from PIL import Image

st.title("Image Caption Generator")

uploaded = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with open("temp.jpg","wb") as f:
        f.write(uploaded.getbuffer())

    caption = generate_caption("temp.jpg")

    st.success("Generated Caption:")
    st.write(caption)
