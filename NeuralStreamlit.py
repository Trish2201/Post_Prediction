
import streamlit as st
import pandas as pd
import re
import tensorflow as tf
import joblib
from PIL import Image
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import gc
from deepface import DeepFace
from scipy.special import inv_boxcox
from transformers import BertTokenizer, TFBertModel
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import tempfile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import boto3

rekognition = boto3.client('rekognition', aws_access_key_id='AKIAZQ3DQFYCFY4CGU6C', aws_secret_access_key='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', region_name='us-east-1')




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
    large_variables = ['df_combined', 'bert_embeddings', 'object_features', 'emotion_features', 'processed_image', 'flattened_features', 'combined_features', 'image_features', 'tabular_data']

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


# Modify the `download_and_preprocess_image` function accordingly to handle image loading for feature extraction

def download_and_preprocess_image(url, target_size=(224, 224)):
    """Download an image from a URL and resize it."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.resize(target_size)
            return img
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None


def extract_emotion_features(image):
    try:
        # Save the PIL Image to a temporary file for analysis
        with tempfile.NamedTemporaryFile(suffix='.jpg', mode='wb', delete=True) as tmp_file:
            image.save(tmp_file, format='JPEG')
            tmp_file.flush()  # Ensure data is written to disk
            
            # Perform emotion analysis using DeepFace
            analysis = DeepFace.analyze(img_path=tmp_file.name, actions=['emotion'], enforce_detection=False)
            
            # Check if the analysis result is in the expected format
            if isinstance(analysis, list) and len(analysis) > 0 and 'emotion' in analysis[0]:
                emotions = analysis[0]['emotion']
                # Convert emotion percentages into a list in a consistent order
                emotion_features = [emotions.get(emotion, 0) for emotion in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]]
                return emotion_features
            else:
                print("Unexpected format in emotion analysis:", analysis)
                return [0] * 7  # Return a default list of zeros if the format is unexpected
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        return [0] * 7  # Return a default list of zeros in case of errors

            
def extract_object_features(image_input):
    num_features = 500  # Define the target number of features
    
    try:
        # Check if the input is a file path or a PIL Image
        if isinstance(image_input, str):
            # It's a file path, load the image as before
            img = image.load_img(image_input, target_size=(224, 224))
        elif isinstance(image_input, Image.Image):
            # It's a PIL Image object, directly use it
            img = image_input.resize((224, 224))
        else:
            raise ValueError("Invalid image input type.")
        
        img_array = image.img_to_array(img)
        img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

        # Obtain feature representations using the feature extractor
        features = feature_extractor.predict(img_preprocessed)
        flattened_features = features.flatten()

        # Adjust the number of features to the desired target size
        if flattened_features.size > num_features:
            return flattened_features[:num_features]
        else:
            return np.pad(flattened_features, (0, num_features - flattened_features.size), 'constant')
    except Exception as e:
        print(f"Error extracting object features: {e}")
        return np.zeros(num_features)




def preprocess_and_extract_features(image_path, datagen, feature_extractor, extract_emotion_features, extract_object_features):
    """
    Preprocesses the uploaded image, extracts emotion and object features,
    and combines them into a single feature vector.

    Parameters:
    - image_path: Path to the image file.
    - datagen: Data generator for image augmentation (if necessary).
    - feature_extractor: Pre-trained model for object feature extraction.
    - extract_emotion_features: Function to extract emotion features.
    - extract_object_features: Function to extract object features.

    Returns:
    - A numpy array containing the combined features.
    """

    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_preprocessed = preprocess_input(np.expand_dims(img_array, axis=0))

    # Extract object features using the feature extractor model
    object_features = feature_extractor.predict(img_preprocessed).flatten()

    # Ensure object features have the correct length (500)
    num_object_features = 500
    if len(object_features) > num_object_features:
        object_features = object_features[:num_object_features]
    else:
        object_features = np.pad(object_features, (0, num_object_features - len(object_features)), 'constant', constant_values=0)

    # Extract emotion features using DeepFace
    emotion_features = extract_emotion_features(image_path)  # Adjusted to accept file path

    # Combine object and emotion features
    combined_features = np.concatenate([object_features, emotion_features])

    return combined_features

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
# feature_extractor = ResNet50(weights='imagenet', include_top=False)
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
#     )

# Load models and transformers
@st.cache_resource(ttl=24*3600)
def load_resources():
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    feature_extractor = joblib.load("feature_extractor.pkl")
    model = tf.keras.models.load_model("neural_network_model.h5")
    #model = tf.saved_model.load('saved_model/my_model_trt', tags=[tag_constants.SERVING])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder1.pkl")  
    text_vectorizer = joblib.load("tfidf_vectorizer.pkl")
    benchmark = joblib.load("benchmark.pkl")
    train_columns = joblib.load("train_columns.pkl")
    boxcox_lambda = joblib.load("boxcox_lambda.pkl")
    feature_extractor = joblib.load("feature_extractor.pkl")
    return bert_model, feature_extractor, model, tokenizer, scaler, encoder, text_vectorizer, benchmark, train_columns, boxcox_lambda

bert_model, feature_extractor, model, tokenizer, scaler, encoder, text_vectorizer, benchmark, train_columns, boxcox_lambda = load_resources()

# Streamlit app
st.title("Hook and Thumbnail A/B Predictor ")


import streamlit as st

# Input section with clear labeling and unique layout
with st.form("input_form"):
    col1, col2 = st.columns(2)
    post_title_default = ""

    with col1:
        post_type = st.selectbox("Select Post Type", ["IG reel", "IG carousel", "IG image"])
        duration = st.number_input("Enter Duration in Seconds", min_value=0, value=30, step=1)
        post_description_container = st.empty()
        #if post_type == "IG image":
        post_description = post_description_container.text_input("Enter Post Copy", help = "caption of the image post displayed")
            #st.write(post_description)
        # else:
        #     post_description = None
        uploaded_image = st.file_uploader("Upload Thumbnail", type=["jpg", "jpeg"], help="Image should be in JPG or JPEG format (not PNG).")

        if st.form_submit_button("Populate Title"):
            if uploaded_image is not None:
                # To see details
                # st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
                # st.write("")

                # Convert the file to an image
                image = Image.open(uploaded_image)

                # Convert to bytes
                img_buffer = BytesIO()
                image.save(img_buffer, format='JPEG')  # Adjust format as needed
                img_bytes = img_buffer.getvalue()

                def analyze_image(image_bytes):
                    response = rekognition.detect_text(
                        Image={'Bytes': image_bytes}
                    )
                    return response

                result = analyze_image(img_bytes)
                post_title_default = ""
                for text_block in result['TextDetections']:
                    if text_block['Type'] == 'LINE':
                        post_title_default += text_block['DetectedText'] + ' '

    with col2:
        relevancy_score = st.slider("Select Google Trend Score", 0.0, 1.0, 0.5, help = "You can keep it 0.5 if the value on the site looks unreliable")

        # Create sliders for each cell in the table
        ig_reel_value = st.slider("Reel Attention Score", min_value=0.0, max_value=1.5, value=1.0, step=0.1)
        ig_carousel_value = st.slider("Carousel Attention Score", min_value=0.0, max_value=1.5, value=1.0, step=0.1)
        ig_image_value = st.slider("Image Attention Score", min_value=0.0, max_value=1.5, value=1.0, step=0.1)


    title = st.text_input("Enter Post Title", help = "any title on the video displayed", value  = post_title_default)
    hook = st.text_area("Enter Post Hook", help="A hook is an attention-grabbing snippet words said in the first 3-5 seconds.")

    # # Checkbox to enable or disable the keyword difficulty slider
    # enable_slider = st.checkbox("Check if the post has any of the following features - Handwritten BSU, Veesual, Tweet style header, Background music ", value=False)
    # checkbox = st.checkbox("Check if the post has any of the following features - Worded titles, Context driven hook, Gary centric content, Niche audience, Advertising book/platform accounts, Long hook, Visually busy title", value=False)
    # culture_checkbox = st.checkbox("Check if the post was released to tap into a culture moment", value=False)

    st.write("Please select if any of the mentioned features apply to your post:")

    # Initialize the variables to False
    good_features = False
    bad_features = False
    cultural_relevance = False

    with st.expander("Ideal Features"):
        if st.checkbox("Handwritten BSU", help="Check if the post includes handwritten visual elements.") or \
        st.checkbox("Veesual", help="Check if the post is a moving image like post.") or \
        st.checkbox("Tweet style header", help="Check if the header is twitter style.") or \
        st.checkbox("Factual/Specific Hook", help="Check if the hook is immediately diving into helpful content.") or \
        st.checkbox("Background music", help="Check if there is accompanying music."):
            good_features = True

    with st.expander("Sub-Optimal Features"):
        if st.checkbox("Worded titles", help="Check if the titles are wordy or verbose.") or \
        st.checkbox("Context driven hook", help="Check if the hook is designed around the context of the content instead of direct punchline.") or \
        st.checkbox("Gary centric content", help="Check if the content is centered too much around Gary.") or \
        st.checkbox("Niche audience", help="Check if the content targets a specific audience.") or \
        st.checkbox("Advertising book/platform accounts", help="Check if it has an appeal of promotional content.") or \
        st.checkbox("Long hook", help="Check if the hook is longer than usual or very worded.") or \
        st.checkbox("Visually busy title", help="Check if the title has a lot of visual elements."):
            bad_features = True

    with st.expander("Cultural Moment"):
        if st.checkbox("The post was released to tap into a culture moment", help = "Do not check this if you have given a 0.8+ score on google trend field already") or \
        st.checkbox("Green Screen", help="Check if the green screen was related to a relevant cultural topic.") :
            cultural_relevance = True
    
    # Store a value based on checkbox selection for later use
    slider_multiplier = 1.5 if good_features else 1
    culture_multiplier = 0.7 if bad_features else 1
    checkbox_multiplier = 1.5 if cultural_relevance else 1

    #refresh_button = st.form_submit_button("Refresh Page", help = "If you select above")
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
            "Title": [f"{title} {hook} {post_description}" if post_description else f"{title} {hook}"],
        }
    )

   # df["Duration (sec)"] = df["Duration (sec)"].apply(lambda duration: 60 - (duration - 60) * 0.10 if duration > 60 else duration)
    df["Duration (sec)"] = np.minimum(df["Duration (sec)"], 45)


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
    prediction_columns = [col for col in train_columns if col != 'Reach'] 
    df = df.reindex(columns=prediction_columns, fill_value=0)
    tabular_data = df.values.reshape(1, -1)

    
    # In the Streamlit app, when predicting
    combined_text = (title if title else "") + " " + (hook if hook else "")

    # Calculate sentiment polarity and subjectivity
    sentiment_polarity = get_sentiment_polarity(combined_text)
    sentiment_subjectivity = get_sentiment_subjectivity(combined_text)

    # Add these values to your dataframe before scaling and encoding
    df['Sentiment_Polarity'] = sentiment_polarity
    df['Sentiment_Subjectivity'] = sentiment_subjectivity

    # Add bert embeddings
    bert_embeddings = get_bert_embeddings(combined_text)
    bert_embeddings = bert_embeddings.reshape(1, -1)

    # Add and normalize numerical columns
    numeric_cols = ["Duration (sec)", "Sentiment_Polarity", "Sentiment_Subjectivity"
                    ]

    # Now you can safely apply imputation
    imputer = SimpleImputer(strategy='mean')

    for col in numeric_cols:
        df[col] = df[col].apply(clean_numeric)

    for col in numeric_cols:
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    scaler = MinMaxScaler()
    for col in numeric_cols:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))


    # Placeholder size for combined features
    placeholder_size = 507

    # Placeholder for image features
    placeholder_image_features = np.zeros((1, placeholder_size))

    if uploaded_image is not None:
        with st.spinner('Processing image...'):
            try:
                # Convert uploaded image to a PIL Image
                img = Image.open(uploaded_image).convert('RGB')
                
                # Emotion feature extraction expects a PIL Image
                emotion_features = extract_emotion_features(img)  # Ensure this is adapted to use PIL Image directly
                #st.write("Emotion features detected:", emotion_features if np.sum(emotion_features) != 0 else "No emotion features detected.")
                
                # print("--- REKOG START ----")
                # img_buffer = BytesIO()
                # img.save(img_buffer, format='JPEG') 
                # img_bytes = img_buffer.getvalue()

                # response = rekognition.detect_text(Image={"Bytes": img_bytes})
                # print("--- REKOG DONE ----")
                # print(response)

                # For object feature extraction, save the PIL Image to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', mode='wb', delete=False) as tmp_file:
                    img.save(tmp_file, format='JPEG')
                    tmp_file.flush()  # Ensure data is written
                    tmp_file_path = tmp_file.name
                    
                    # Extract object features
                    object_features = extract_object_features(tmp_file_path)  # Ensure this is ready for file paths
                    #st.write("Object features detected:", object_features if np.any(object_features != 0) else "No object features detected.")

                # Combine features
                combined_features = np.concatenate([object_features, emotion_features])
                image_features = combined_features.reshape(1, -1)  # Reshape for compatibility with prediction inputs
                st.success('Image processed successfully.')

            except Exception as e:
                st.error(f"Error in processing the image: {e}")
                image_features = placeholder_image_features
    else:
        image_features = placeholder_image_features



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
    if post_type == 'IG reel':
        final_estimated_reach = estimated_reach * multiple * multiple_2 * ig_reel_value * slider_multiplier * culture_multiplier * checkbox_multiplier
        final_estimated_reach1 = estimated_reach * multiple * 0.8 * ig_reel_value
    elif post_type == 'IG carousel':
        final_estimated_reach = estimated_reach * multiple * multiple_2 * ig_carousel_value * slider_multiplier * culture_multiplier * checkbox_multiplier
        final_estimated_reach1 = estimated_reach * multiple * 0.8 * ig_carousel_value
    else:
        final_estimated_reach = estimated_reach * multiple * multiple_2 * ig_image_value * slider_multiplier * culture_multiplier * checkbox_multiplier
        final_estimated_reach1 = estimated_reach * multiple * 0.8 * ig_image_value


    # Call the function to display the output
    display_custom_visual_prediction(final_estimated_reach, final_estimated_reach1, benchmark)
    
    # Calculate 10% of the final_estimated_reach
    reach_variation = 150000

    # Calculate the lower and upper bounds
    lower_bound = final_estimated_reach - reach_variation
    upper_bound = final_estimated_reach + reach_variation

    # Correctly format the numbers to remove any trailing decimals
    formatted_final_estimated_reach = f"{final_estimated_reach:,.0f}"
    formatted_lower_bound = f"{lower_bound:,.0f}"
    formatted_upper_bound = f"{upper_bound:,.0f}"

    # Simplified formatting for clarity
    markdown_string = f"""
            <style>
                .metric-container {{
                    border: 2px solid #4CAF50; /* Subtle emphasis with color */
                    border-radius: 8px;
                    padding: 15px 20px;
                    background-color: #f9f9f9; /* Soft background for contrast */
                    display: inline-block; /* Adjust as needed */
                }}
                .metric-label {{
                    font-size: 20px; /* Clear, readable font size */
                    font-weight: bold;
                    color: #333; /* Ensuring good readability */
                }}
                .metric-value {{
                    font-size: 20px; /* Keeping important numbers prominent */
                    font-weight: bold;
                    color: #4CAF50; /* Adding a bit of color for emphasis */
                    margin-left: 5px; /* Spacing for aesthetics */
                }}
                .metric-range {{
                    font-size: 18px; /* Slightly smaller font for the range */
                    color: #777; /* Less emphasis on the range */
                    margin-left: 5px; /* Aesthetic spacing */
                    font-style: italic; /* Differentiation */
                }}
            </style>
            <div class="metric-container">
                <span class="metric-label">Reach Prediction:</span>
                <span class="metric-value">{final_estimated_reach:,.0f}</span>
                <span class="metric-range">([{lower_bound:,.0f}-{upper_bound:,.0f}])</span>
            </div>
            """

    # This is where you would use the markdown_string with your streamlit code, like:
    st.markdown(markdown_string, unsafe_allow_html=True)

# Ensure to call clear_memory() where appropriate to manage memory usage
clear_memory()

# Footer
st.write("---")
st.write("Powered by Data Science")


