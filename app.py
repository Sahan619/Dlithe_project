import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Load the trained model
model = tf.keras.models.load_model('model.h5')
st.set_page_config(
    page_title="Plant Disease Detection App",
    page_icon="🌿",
)
background_image_url = "https://img.freepik.com/free-vector/hand-painted-watercolor-nature-background_23-2148934719.jpg"

# Define the labels
labels = [
    "Apple___Infected",
    "Apple___Healthy",
]

# Set Streamlit title and description
st.title('Plant Leaf Detection')
st.write('Upload an image to classify it as infected or healthy apple.')

# File uploader widget
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the uploaded image
    img_height, img_width = 224, 224
    img = tf.keras.preprocessing.image.load_img(uploaded_file, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions
    predicted_probabilities = model.predict(img_array)[0]
    predicted_label_index = np.argmax(predicted_probabilities)
    predicted_label = labels[predicted_label_index]

    # Display predictions
    st.subheader('Prediction:')
    st.write(f'Predicted Label: {predicted_label}')
    #st.write(f'Predicted Probabilities: {predicted_probabilities}')




