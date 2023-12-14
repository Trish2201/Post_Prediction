import streamlit as st
import pandas as pd
import numpy as np
import re
import tensorflow as tf
import joblib
from PIL import Image
import random
import io
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc

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

def clear_memory():
    # List of large variables to clear
    large_variables = ['df', 'augmented_images', 'image_features', 'placeholder_image_features', 'processed_image']

    for var in large_variables:
        if var in globals():
            del globals()[var]
    
    # Garbage collection to free up memory
    gc.collect()



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


def download_and_preprocess_image(image_content, target_size=(224, 224)):
    if image_content:
        try:
            # Convert the binary content to a bytes stream
            image_stream = io.BytesIO(image_content)
            image = Image.open(image_stream)
            image = image.resize(target_size)
            img_array = img_to_array(image)
            img_array = img_array / 255.0  # Scale pixel values
            return img_array
        except IOError as e:
            print(f"Error processing the image: {e}")
            return None
    else:
        return None

def preprocess_and_extract_features(processed_image, datagen, feature_extractor):
    if processed_image is not None:
        augmented_images = [datagen.random_transform(processed_image) for _ in range(5)]
        augmented_features = [feature_extractor.predict(np.expand_dims(img, axis=0)) for img in augmented_images]
        
        # Calculate the mean of the features
        mean_features = np.mean(augmented_features, axis=0)
        
        # Flatten the features and reshape to match the model's expected input shape
        flattened_features = mean_features.flatten()
        if flattened_features.size >= 500:  # Ensure there are enough elements
            flattened_features = flattened_features[:500]  # Truncate or reshape as needed
        else:
            # If not enough features, pad with zeros
            flattened_features = np.pad(flattened_features, (0, 500 - flattened_features.size), 'constant')
        
        return flattened_features
    else:
        # Return an array of zeros with the shape (500,) if no image is processed
        return np.zeros((500,))



# Initialize the feature extractor (example with VGG16)
feature_extractor = ResNet50(weights='imagenet', include_top=False)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
    )



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
uploaded_image = st.file_uploader("Upload thumbnail", type=["jpg", "jpeg"])

# New inputs for Relevancy Score and Multiple
relevancy_score = st.number_input("Relevancy Score", min_value=0.0, value = 100.0, format="%.2f") # default value entering = 100
multiple = st.number_input("Multiple", min_value=0.0, value = 0.01, format="%.2f") # default value entering = 0.01


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
    prediction_columns = [col for col in train_columns if col != 'Reach'] # removing 'Reach' from train_columns to limit it to input features
    df = df.reindex(columns=prediction_columns, fill_value=0)
    tabular_data = df.values.reshape(1, -1)

    # initializing image_features
    image_features = None
    placeholder_image_features = np.zeros((1, 500)) # case

    # Process the uploaded image and extract features
    if uploaded_image is not None:
        with st.spinner('Processing image...'):
            processed_image = download_and_preprocess_image(uploaded_image.getvalue())
            if processed_image is not None:
                image_features = preprocess_and_extract_features(processed_image, datagen, feature_extractor)
                st.success('Image processed successfully.')
                # Reshape image_features to match the model's expected input shape
                if len(image_features.shape) == 1:
                    image_features = np.expand_dims(image_features, axis=0)
            else:
                st.error('Error in processing the image.')
                image_features = placeholder_image_features
    else:
        # Use the placeholder as the image features when no image is uploaded
        image_features = placeholder_image_features


    # Check if image_features is defined, then prepare prediction inputs accordingly
    if image_features is not None:
        # Reshape image_features to have the same first dimension as tabular_data
        if len(image_features.shape) == 1:  # If image_features is 1D
            image_features = np.expand_dims(image_features, axis=0)
        # Combine image features with tabular data for prediction
        prediction_inputs = [image_features, tabular_data]
    else:
        # Use only tabular_data for prediction
        prediction_inputs = [tabular_data]


    # Prediction
    prediction = model.predict(prediction_inputs)
    
    # Display the prediction
    estimated_reach = int(prediction[0])

    # Adjust the final estimated reach based on Relevancy Score and Multiple
    final_estimated_reach = estimated_reach * relevancy_score * multiple
 
    # # Display the final prediction
    st.write(f"Final Estimated Reach: {final_estimated_reach}")
    st.write(
        f"Final Estimated Reach based on Benchmark: {((final_estimated_reach / int(benchmark)) * 100):.2f}%"
    )
    st.write(f"Benchmark: {int(benchmark)}")

    # Once prediction is done, call clear_memory to free up space
    clear_memory()

# Footer
st.write("---")
st.write("Powered by Team Gary")
