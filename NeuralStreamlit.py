
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
import cv2
from deepface import DeepFace
from scipy.special import inv_boxcox
from transformers import BertTokenizer, TFBertModel
from textblob import TextBlob



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


# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(text):
    # Check if the text is empty or null
    if text is None or text.strip() == "":
        text = "no text"

    # Generate BERT embeddings
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = bert_model(inputs["input_ids"])
    return outputs.last_hidden_state[:, 0, :].numpy()



def download_and_preprocess_image(image_content, target_size=(224, 224)):
    if image_content:
        try:
            image_stream = io.BytesIO(image_content)
            image = Image.open(image_stream)
            image = image.resize(target_size)
            img_array = img_to_array(image)
            img_array = img_array / 255.0

            emotion_features = extract_emotion_features(image)
            object_features = extract_object_features(image)

            return img_array, emotion_features, object_features
        except IOError as e:
            print(f"Error processing the image: {e}")
            return None, [0] * 7, [0] * 10
    else:
        return None, [0] * 7, [0] * 10

def preprocess_and_extract_features(processed_image, emotion_features, object_features, datagen, feature_extractor):
    if processed_image is not None:
        augmented_images = [datagen.random_transform(processed_image) for _ in range(5)]
        augmented_features = [feature_extractor.predict(np.expand_dims(img, axis=0)) for img in augmented_images]
        mean_features = np.mean(augmented_features, axis=0).flatten()
        combined_features = np.concatenate((mean_features, emotion_features, object_features))
        
        # Resize the feature vector to a fixed size if necessary
        target_size = 500  # Adjust as needed
        if combined_features.size >= target_size:
            return combined_features[:target_size]
        else:
            return np.pad(combined_features, (0, target_size - combined_features.size), 'constant')
    else:
        return np.zeros((500,))


def extract_emotion_features(image):
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['emotion'])
        return [analysis["emotion"][emotion] for emotion in analysis["emotion"]]
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        return [0] * 7

def extract_object_features(image):
    # Placeholder for object detection logic
    num_object_features = 10
    return [0] * num_object_features  

# def normalize_embeddings(embeddings):
#     norm = np.linalg.norm(embeddings)
#     if norm == 0:
#        return embeddings
#     return embeddings / norm



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
boxcox_lambda = joblib.load("boxcox_lambda.pkl")

# Streamlit app
st.title("Instagram Post Reach Predictor ")

# Inputs
post_type = st.selectbox("Post Type", ["IG reel", "IG carousel", "IG image"])
duration = st.number_input("Duration (sec)", 0)

title = st.text_input("Title")
transcription = st.text_input("Hook")


# Function to detect numbers in text
def contains_numbers(text):
    return bool(re.search(r'\d', text))

# Detect numbers in the title
if contains_numbers(title):
    multiple_2 = 1.2
else:
    multiple_2 = 1


# Image upload
uploaded_image = st.file_uploader("Upload thumbnail", type=["jpg", "jpeg"])

# New inputs for Relevancy Score and Multiple
relevancy_score = st.number_input("Choose the cultural relevance score between 0 to 1", min_value=0.0, value = 0.5, format="%.2f") # default value entering = 100


# Define multiple based on relevancy_score
if relevancy_score <= 0.3:
    multiple = 0.8
elif 0.3 < relevancy_score <= 0.5:
    multiple = 0.9
elif 0.5 < relevancy_score <= 0.6:  
    multiple = 1
elif 0.6 < relevancy_score <= 0.8:
    multiple = 1.2
elif 0.8 < relevancy_score <= 1:
    multiple = 1.5
else:
    multiple = 1 

def get_sentiment_polarity(text):
    """
    Returns the sentiment polarity of the given text.
    Polarity is a float within the range [-1.0, 1.0].
    """
    return TextBlob(text).sentiment.polarity

def get_sentiment_subjectivity(text):
    """
    Returns the sentiment subjectivity of the given text.
    Subjectivity is a float within the range [0.0, 1.0] where 0.0 is very objective and 1.0 is very subjective.
    """
    return TextBlob(text).sentiment.subjectivity


# Predict Reach
if st.button("Predict Reach"):
    # Create a dataframe from inputs
    df = pd.DataFrame(
        {
            "Post type": [post_type],
            "Duration (sec)": [duration],
            "Title": [title + " " + transcription],
        }
    )

    # Convert 'Title' using TF-IDF vectorizer
    title_matrix = text_vectorizer.transform(df["Title"])
    df_title = pd.DataFrame(
        title_matrix.toarray(), columns=text_vectorizer.get_feature_names_out()
    )

    # Merge the TF-IDF dataframe with the main dataframe
    df = pd.concat([df.drop(columns=["Title"]), df_title], axis=1)

    # Encode categorical features
    categorical_cols = ["Post type"] # "Publish Day", "Time of day"
    df_encoded = pd.DataFrame(
        encoder.transform(df[categorical_cols]),
        columns=encoder.get_feature_names_out(),
    )
    df = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)

    # Ensure df has the same columns as during model training, minus 'Reach'
    prediction_columns = [col for col in train_columns if col != 'Reach'] # removing 'Reach' from train_columns to limit it to input features
    df = df.reindex(columns=prediction_columns, fill_value=0)
    tabular_data = df.values.reshape(1, -1)

    # BERT Embeddings
    
    # In the Streamlit app, when predicting
    combined_text = (title if title else "") + " " + (transcription if transcription else "")
    # Calculate sentiment polarity and subjectivity
    sentiment_polarity = get_sentiment_polarity(combined_text)
    sentiment_subjectivity = get_sentiment_subjectivity(combined_text)

    # Add these values to your dataframe before scaling and encoding
    df['Sentiment_Polarity'] = sentiment_polarity
    df['Sentiment_Subjectivity'] = sentiment_subjectivity
    bert_embeddings = get_bert_embeddings(combined_text)
    bert_embeddings = bert_embeddings.reshape(1, -1)

    # Normalize BERT embeddings
    #bert_embeddings_normalized = normalize_embeddings(bert_embeddings)
    #bert_embeddings_normalized = bert_embeddings_normalized.reshape(1, -1)


    # Scale numeric features
    numeric_cols = ["Duration (sec)", "Sentiment_Polarity", "Sentiment_Subjectivity"]
    
    df[numeric_cols] = scaler.transform(df[numeric_cols])



    # initializing image_features
    image_features = None
    placeholder_image_features = np.zeros((1, 500)) # case

    if uploaded_image is not None:
        with st.spinner('Processing image...'):
            processed_image, emotion_features, object_features = download_and_preprocess_image(uploaded_image.getvalue())
            if processed_image is not None:
                image_features = preprocess_and_extract_features(processed_image, emotion_features, object_features, datagen, feature_extractor)
                st.success('Image processed successfully.')
            else:
                st.error('Error in processing the image.')
                image_features = np.zeros((500,))
    else:
        image_features = np.zeros((500,))


    # Check if image_features is defined, then prepare prediction inputs accordingly
    if image_features is not None:
        # Reshape image_features to have the same first dimension as tabular_data
        if len(image_features.shape) == 1:  # If image_features is 1D
            image_features = np.expand_dims(image_features, axis=0)
        # Combine image features with tabular data for prediction
        prediction_inputs = [image_features, tabular_data, bert_embeddings]
    else:
        # Use only tabular_data for prediction
        prediction_inputs = [tabular_data, bert_embeddings]

    print("Image: ",type(image_features),type(image_features[0]),type(image_features[0][0]))
    print("Tabular: ",type(tabular_data),type(tabular_data[0]),type(tabular_data[0][0]))


    # Prediction
    prediction = model.predict(prediction_inputs)

    # Apply inverse Box-Cox transformation
    if np.isnan(prediction).any():
        st.error("Prediction contains NaN values.")
    prediction = inv_boxcox(prediction, boxcox_lambda)
    
    # Display the prediction
    estimated_reach = int(prediction[0])


    # Adjust the final estimated reach based on Relevancy Score and Multiple
    final_estimated_reach = estimated_reach*multiple*multiple_2*3
 
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
st.write("For reels above 60 seconds, just enter 60 in the Duration box for now")

