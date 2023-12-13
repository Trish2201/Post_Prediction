import streamlit as st
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import joblib
from PIL import Image
import random
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Custom CSS
st.markdown(
    """
    <style>
        .reportview-container {
            background: #D8BFD8;
        }
        .main {
           background: #D8BFD8;
           color: black;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Function to extract emojis
def extract_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.findall(text)


# Function to convert HEX to RGB
def hex_to_rgb(value):
    value = value.lstrip("#")
    length = len(value)
    return tuple(
        int(value[i : i + length // 3], 16) for i in range(0, length, length // 3)
    )

# Function to preprocess images and extract features
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features_from_image(image):
    feature_extractor = ResNet50(weights='imagenet', include_top=False)
    features = feature_extractor.predict(np.expand_dims(image, axis=0))
    return features.flatten()



# Load models and transformers
model = tf.keras.models.load_model("neural_network_model.h5")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder1.pkl")
text_vectorizer = joblib.load("tfidf_vectorizer.pkl")
benchmark = joblib.load("benchmark.pkl")
train_columns = joblib.load("train_columns.pkl")

# Streamlit app
st.title("Instagram Post Reach Predictor ")

# Inputs
post_type = st.selectbox("Post Type", ["IG reel", "IG carousel", "IG image"])
duration = st.number_input("Duration (sec)", 0)
time_of_day = st.selectbox(
    "Time of Day",
    [
        "4AM",
        "5AM",
        "6AM",
        "7AM",
        "8AM",
        "9AM",
        "10AM",
        "11AM",
        "12PM",
        "1PM",
        "2PM",
        "3PM",
        "4PM",
        "5PM",
        "6PM",
        "7PM",
        "8PM",
        "9PM",
        "10PM",
        "11PM",
    ],
)
publish_day = st.selectbox(
    "Publish Day",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
)
title = st.text_input("Title")
transcription = st.text_input("Hook")

# Add color pickers
title_text_color_hex = st.color_picker(
    "Title Text Color", "#000000"
)  # Default color is black
title_background_color_hex = st.color_picker(
    "Title Background Color", "#FFFFFF"
)  # Default color is white

# Convert HEX colors to RGB
title_text_color_rgb = hex_to_rgb(title_text_color_hex)
title_background_color_rgb = hex_to_rgb(title_background_color_hex)

# New input widgets for the two questions
contains_number = st.selectbox("Contains Number in Title?", ["Yes", "No"], index=1)
multiple_fonts = st.selectbox("Multiple Font Colors Detected in Title?", ["Yes", "No"], index=1)

# Convert the user's answers to 1 or 0
contains_number = 1 if contains_number == "Yes" else 0
multiple_fonts = 1 if multiple_fonts == "Yes" else 0

# Image upload
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# New inputs for Relevancy Score and Multiple
relevancy_score = st.number_input("Relevancy Score", min_value=0.0, value = 100.0, format="%.2f")
multiple = st.number_input("Multiple", min_value=0.0, value = 0.01, format="%.2f") # default value entering


# Predict Reach
if st.button("Predict Reach"):
    # Create a dataframe from inputs
    df = pd.DataFrame(
        {
            "Post type": [post_type],
            "Duration (sec)": [duration],
            "Time of day": [time_of_day],
            "Publish Day": [publish_day],
            "Title": [title + " " + transcription],
            "Word Count": [len(title.split())],
            "Title Text Color_R": [title_text_color_rgb[0]],
            "Title Text Color_G": [title_text_color_rgb[1]],
            "Title Text Color_B": [title_text_color_rgb[2]],
            "Title Background Color_R": [title_background_color_rgb[0]],
            "Title Background Color_G": [title_background_color_rgb[1]],
            "Title Background Color_B": [title_background_color_rgb[2]],
            "Contains Number in Title?": [contains_number],
            "Multiple Font Colors Detected in Title?": [multiple_fonts],
        }
    )

    # Scale numeric features
    numeric_cols = ["Duration (sec)", "Word Count", "Contains Number in Title?", "Multiple Font Colors Detected in Title?"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Convert 'Title' using TF-IDF vectorizer
    title_matrix = text_vectorizer.transform(df["Title"])
    df_title = pd.DataFrame(
        title_matrix.toarray(), columns=text_vectorizer.get_feature_names_out()
    )

    # Merge the TF-IDF dataframe with the main dataframe
    df = pd.concat([df.drop(columns=["Title"]), df_title], axis=1)

    # Encode categorical features
    categorical_cols = ["Post type", "Publish Day", "Time of day"]
    df_encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(),
    )
    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    # Ensure df has the same columns as during model training, minus 'Reach'
    prediction_columns = [col for col in train_columns if col != 'Reach']
    df = df.reindex(columns=prediction_columns, fill_value=0)
    tabular_data = df.values.reshape(1, -1)

    # train_columns.remove("Reach")
    # df = df[train_columns]

    # Process the uploaded image
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        processed_image = preprocess_image(image)
        image_features = extract_features_from_image(processed_image)
        image_features = image_features.reshape(1, -1)  # Reshape for compatibility with model input
    else:
        # If no image is uploaded, use a zero array as a placeholder
        image_features = np.zeros((1, 500))  # Assuming 500 features from image

    # Combine image features with tabular data for prediction
    prediction_inputs = [image_features, tabular_data]

    # Prediction
    prediction = model.predict(prediction_inputs)
    
    # Display the prediction
    estimated_reach = int(prediction[0])

    # Adjust the final estimated reach based on Relevancy Score and Multiple
    final_estimated_reach = estimated_reach * relevancy_score * multiple

    # st.write(f"Estimated Reach: {estimated_reach}")
    # st.write(
    #     f"Estimated Reach based on Benchmark: {((estimated_reach / int(benchmark)) * 100):.2f}%"
    # )
    # st.write(f"Benchmark: {int(benchmark)}")
 
    # # Display the final prediction
    st.write(f"Final Estimated Reach: {final_estimated_reach}")
    st.write(
        f"Final Estimated Reach based on Benchmark: {((final_estimated_reach / int(benchmark)) * 100):.2f}%"
    )
    st.write(f"Benchmark: {int(benchmark)}")

# Footer
st.write("---")
st.write("Powered by Team Gary")
