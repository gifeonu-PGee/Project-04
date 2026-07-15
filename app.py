import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Force CPU only — avoids CUDA errors on Streamlit Cloud
tf.config.set_visible_devices([], 'GPU')

IMG_SIZE = (160, 160)

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾"
)

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("project03_mobilenet_cats_dogs.keras")

model = load_trained_model()

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image, and the AI will predict whether it is a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    resized_image = image.resize(IMG_SIZE)
    image_array = np.array(resized_image, dtype=np.float32)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = float(model.predict(image_array, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "Dog 🐶"
        confidence = prediction
    else:
        label = "Cat 🐱"
        confidence = 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence:.2%}")

    if confidence >= 0.95:
        st.success("The model is highly confident.")
    elif confidence >= 0.80:
        st.info("The model is reasonably confident.")
    else:
        st.warning("The model is uncertain. Try another clear image.")
