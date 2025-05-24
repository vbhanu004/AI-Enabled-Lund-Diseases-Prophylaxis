import streamlit as st
import tensorflow as tf
import numpy as np


model=tf.keras.models.load_model("E:\Lung Disease Prediction using CNN\model.hdf5")
target_size=(256,256)
class_names=["Lung_Opacity","Normal","Covid","Viral Pneumonia"]

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = tf.image.resize(image, target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Streamlit app
def main():
    st.set_page_config("COVID-19 Detection")

    st.title("Lung Disease Prediction")

    st.sidebar.title("Upload an Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = tf.image.decode_image(uploaded_file.read(), channels=3)
        st.image(image.numpy(), caption="Uploaded Image", use_column_width=True)
        st.write("")

        if st.button("Predict"):
            with st.spinner("Predicting..."):
                img_array = preprocess_image(image)
                prediction = model.predict(img_array)
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

            st.success(f"Predicted Class: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()