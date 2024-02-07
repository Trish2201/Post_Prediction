
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc
import cv2
from deepface import DeepFace
from scipy.special import inv_boxcox
from transformers import BertTokenizer, TFBertModel
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer



# # Custom CSS
# st.markdown(
#     """
#     <style>
#         .reportview-container {
#             background: #D8BFD8;
#         }
#         .main {
#            background: #D8BFD8;
#            color: black;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# Custom CSS to inject for a vibrant and professional look
def inject_custom_css():
    st.markdown(
        """
        <style>
            /* Main container settings */
            .reportview-container .main {
                color: #ffffff; /* White text for better contrast on vibrant backgrounds */
                background-color: #1F1F1F; /* Dark background to make the vibrant elements pop */
            }

            /* Utilize nearly full screen width */
            .reportview-container .main .block-container {
                max-width: 95%;
            }

            /* Streamlit's default padding adjustments for a tailored layout */
            .reportview-container .main .block-container {
                padding-top: 2rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 2rem;
            }

            /* Headers with custom font and vibrant colors */
            h1 {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 3rem;
                font-weight: bold;
                color: #FF2E63; /* Vibrant pink */
            }

            h2 {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 2.5rem;
                font-weight: bold;
                color: #08D9D6; /* Vibrant cyan */
            }

            /* Custom button styles with vibrant gradients */
            .stButton>button {
                font-size: 1rem;
                padding: 0.5rem 2.5rem;
                color: #1F1F1F;
                background: linear-gradient(90deg, #FF2E63 0%, #08D9D6 100%);
                border: none;
                border-radius: 5px;
                transition: all 0.3s ease;
                box-shadow: 0px 4px 6px rgba(255, 255, 255, 0.3);
            }

            .stButton>button:hover {
                background: linear-gradient(90deg, #08D9D6 0%, #FF2E63 100%);
                box-shadow: 0px 6px 15px rgba(255, 255, 255, 0.45);
            }

            /* Custom file uploader styling to match the vibrant theme */
            .stFileUploader {
                border: 2px solid #08D9D6;
                border-radius: 5px;
                color: #08D9D6;
            }

            /* Metric styling for standout display */
            .stMetricLabel {
                font-weight: bold;
                color: #FF2E63; /* Matching the vibrant pink color */
            }

            .stMetricValue {
                font-size: 2.5rem;
                font-weight: bold;
                color: #08D9D6; /* Matching the vibrant cyan color */
            }

            /* Footer with a custom font and vibrant color */
            .footer {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                font-size: 0.9rem;
                color: #FF2E63;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Inject custom CSS
inject_custom_css()



def clear_memory():
    # List of large variables to clear
    large_variables = ['df', 'bert_embeddings', 'object_features', 'emotion_features', 'processed_image']

    for var in large_variables:
        if var in globals():
            del globals()[var]
    
    # Garbage collection to free up memory
    gc.collect()


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

# Function to clean numerical variables we will be training on
def clean_numeric(value):
    try:
        if isinstance(value, str):
            value = value.replace(',', '')
        return float(value)
    except ValueError:
        return np.nan  # return NaN for values that cannot be converted to float 
    

# Function to detect numbers in text
def contains_numbers(text):
    return bool(re.search(r'\d', text))

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


# Display the prediction visually with Matplotlib
def display_custom_visual_prediction(final_estimated_reach, final_estimated_reach1, benchmark):
    # Create a figure and a bar plot
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size to match your form size

    # Bar locations
    bar_locs = range(3)

    # Bar heights based on data
    heights = [final_estimated_reach1, benchmark, final_estimated_reach]

    # Bar labels
    labels = ['Average Relevant Reach', 'Benchmark', 'Predicted Reach']

    # Create bars
    bars = ax.bar(bar_locs, heights, color=['#ff9999', '#ffcc99', '#ff9999'])  # Pink shades

    # Add the actual value at the center of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval/2, f'{yval/1e6:.1f}M', ha='center', va='center', color='black', fontsize=10)

    # Add an annotation for the percentage above the benchmark on top of the 'Predicted Reach' bar
    percentage_above = (final_estimated_reach / benchmark - 1) * 100
    # Define colors for positive and negative percentages
    arrow_color = 'green' if percentage_above >= 0 else 'red'

    # Add an annotation for the percentage above the benchmark on top of the 'Predicted Reach' bar
    ax.annotate(f'{percentage_above:.2f}%', 
            xy=(2, final_estimated_reach), 
            xytext=(2, final_estimated_reach + benchmark * 0.1),  # adjust this offset as needed
            arrowprops=dict(facecolor=arrow_color, shrink=0.05), 
            ha='right', 
            fontsize=10)

    # Set the x-axis labels to the bar labels
    ax.set_xticks(bar_locs)
    ax.set_xticklabels(labels, fontsize=10)

    # Set the y-axis label
    ax.set_ylabel('Reach', fontsize=12)

    # Format the y-axis to show 'M' for millions and one decimal place
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

    # Set the y-axis range to go up to 2.5M
    ax.set_ylim(0, 2.5e6)

    # Add a title above the graph
    plt.title('Results', fontsize=14)

    # Customize the grid
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.xaxis.grid(False)  # Turn off the grid for the x-axis

    # Remove the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Display the plot in the Streamlit app
    st.pyplot(fig)


# Initialize the feature extractor (example with VGG16)
feature_extractor = ResNet50(weights='imagenet', include_top=False)
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
    )

# Load models and transformers
@st.cache_resource
def load_resources():
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    feature_extractor = ResNet50(weights='imagenet', include_top=False)
    model = tf.keras.models.load_model("neural_network_model.h5")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder1.pkl")  # Adjust the name if necessary
    text_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    benchmark = joblib.load("benchmark.pkl")
    train_columns = joblib.load("train_columns.pkl")
    boxcox_lambda = joblib.load("boxcox_lambda.pkl")
    return bert_model, feature_extractor, model, tokenizer, scaler, encoder, text_vectorizer, benchmark, train_columns, boxcox_lambda

bert_model, feature_extractor, model, tokenizer, scaler, encoder, text_vectorizer, benchmark, train_columns, boxcox_lambda = load_resources()

# Streamlit app
st.title("Instagram Post Reach Predictor ")

# # Inputs
# post_type = st.selectbox("Post Type", ["IG reel", "IG carousel", "IG image"])
# duration = st.number_input("Duration (sec)", 0)
# title = st.text_input("Title")
# transcription = st.text_input("Hook")
# uploaded_image = st.file_uploader("Upload thumbnail", type=["jpg", "jpeg"])
# relevancy_score = st.number_input("Choose the cultural relevance score between 0 to 1", min_value=0.0, value = 0.5, format="%.2f") # default value entering = 100

# Input section with clear labeling and unique layout
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        post_type = st.selectbox("Select Post Type", ["IG reel", "IG carousel", "IG image"], index=0)
        duration = st.number_input("Enter Duration in Seconds", min_value=0, value=30, step=1)

    with col2:
        relevancy_score = st.slider("Select Cultural Relevance Score", 0.0, 1.0, 0.5)
        uploaded_image = st.file_uploader("Upload Thumbnail", type=["jpg", "jpeg"], help="Image should be in JPG or JPEG format.")

    title = st.text_input("Enter Post Title", help = "any title on the video displayed")
    hook = st.text_area("Enter Post Hook", help="A hook is an attention-grabbing snippet words said in the first 3-5 seconds.")

    submit_button = st.form_submit_button("Predict Reach")


# Detect numbers in the title
if contains_numbers(title):
    multiple_2 = 1.2
else:
    multiple_2 = 1

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
    multiple = 0.5


# Predict Reach
if submit_button:
    # Create a dataframe from inputs
    df = pd.DataFrame(
        {
            "Post type": [post_type],
            "Duration (sec)": [duration],
            "Title": [title + " " + hook],
        }
    )

    # Limit the duration to a maximum of 60 seconds
    df["Duration (sec)"] = min(duration, 90)

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
    combined_text = (title if title else "") + " " + (hook if hook else "")

    # Calculate sentiment polarity and subjectivity
    sentiment_polarity = get_sentiment_polarity(combined_text)
    sentiment_subjectivity = get_sentiment_subjectivity(combined_text)

    # Add these values to your dataframe before scaling and encoding
    df['Sentiment_Polarity'] = sentiment_polarity
    df['Sentiment_Subjectivity'] = sentiment_subjectivity
    bert_embeddings = get_bert_embeddings(combined_text)
    bert_embeddings = bert_embeddings.reshape(1, -1)


    numeric_cols = ["Duration (sec)", "Sentiment_Polarity", "Sentiment_Subjectivity"
                    ]

    # Now you can safely apply imputation
    imputer = SimpleImputer(strategy='mean')


    for col in numeric_cols:
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)

    scaler = MinMaxScaler()
    for col in numeric_cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))


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
    final_estimated_reach1 = estimated_reach*multiple*1.5


    # Call the function to display the output
    display_custom_visual_prediction(final_estimated_reach, final_estimated_reach1, benchmark)

    # Once prediction is done, call clear_memory to free up space
    clear_memory()

# Footer
st.write("---")
st.write("Powered by Team Gary")


