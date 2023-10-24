import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib

# Custom CSS
# Custom CSS
st.markdown("""
    <style>
        .reportview-container {
            background: #D8BFD8;
        }
        .main {
           background: #D8BFD8;
           color: black;
        }
    </style>
    """, unsafe_allow_html=True)



# Extracting emojis
def extract_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    
    return emoji_pattern.findall(text)

# Function to convert HEX to RGB
def hex_to_rgb(value):
    value = value.lstrip('#')
    length = len(value)
    return tuple(int(value[i:i+length//3], 16) for i in range(0, length, length//3))

# Load models and transformers
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
encoder = joblib.load('encoder.pkl')
train_columns = joblib.load('train_columns.pkl')
benchmark = joblib.load('benchmark.pkl')


# Streamlit app
st.title('Instagram Post Reach Predictor 🚀 ')

# Inputs
post_type = st.selectbox('Post Type', ['IG reel', 'IG carousel' ,'IG image'])
duration = st.number_input("Duration (sec)", 0)
time_of_day = st.selectbox('Time of Day', ['4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM', '12PM', '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM'])
publish_day = st.selectbox('Publish Day', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
#description = st.text_area('Description')
title = st.text_input('Title')
# hook = st.text_input('Hook')

# Add color pickers
title_text_color_hex = st.color_picker('Title Text Color', '#000000') # Default color is black
title_background_color_hex = st.color_picker('Title Background Color', '#FFFFFF') # Default color is white

title_text_color_rgb = hex_to_rgb(title_text_color_hex)
title_background_color_rgb = hex_to_rgb(title_background_color_hex)

# New input widgets for the two questions
contains_number = st.selectbox('Contains Number in Title?', ['Yes', 'No'])
multiple_fonts = st.selectbox('Multiple Font Colors Detected in Title?', ['Yes', 'No'])

# Convert the user's answers to 1 or 0
contains_number = 1 if contains_number == 'Yes' else 0
multiple_fonts = 1 if multiple_fonts == 'Yes' else 0

# On pressing the predict button
if st.button('Predict Reach'):
    # Create a dataframe from inputs
    df = pd.DataFrame({
        'Post type': [post_type],
        'Duration (sec)': [duration],
        'Time of day': [time_of_day],
        'Publish Day': [publish_day],
    #    'Description': [description],
        'Title': [title],
        # 'Hook': [hook],
        'Word Count': [len(title.split())],
        'Title Text Color - R': [title_text_color_rgb[0]],
        'Title Text Color - G': [title_text_color_rgb[1]],
        'Title Text Color - B': [title_text_color_rgb[2]],
        'Title Background Color - R': [title_background_color_rgb[0]],
        'Title Background Color - G': [title_background_color_rgb[1]],
        'Title Background Color - B': [title_background_color_rgb[2]],
        'Contains Number in Title?': [contains_number],
        'Multiple Font Colors Detected in Title?': [multiple_fonts],
    })

    # Convert 'Description', 'Title', and 'Hook' using their respective TF-IDF vectorizers
    # description_matrix = vectorizer.transform(df['Description'])
    # df_description = pd.DataFrame(description_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    title_matrix = vectorizer.transform(df['Title'])
    df_title = pd.DataFrame(title_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # hook_matrix = hook_vectorizer.transform(df['Hook'])
    # df_hook = pd.DataFrame(hook_matrix.toarray(), columns=hook_vectorizer.get_feature_names_out())

    # Merge the TF-IDF dataframes with the main dataframe
    df = pd.concat([df, df_title], axis=1) #df_description
    df = df.drop(columns=[ 'Title'])

    # Prediction
    file1 = open('myfile2.txt', 'w')
    # print(len(df.columns.tolist()))
    file1.write(str(df.columns.tolist()))
    file1.close()

    # Ordinal Encode the 'Post type', 'Publish Day', 'Time of day'
    df = encoder.transform(df)

    # Process Emojis
    # df['emojis'] = df['Title'].apply(extract_emojis)
    # df['emojis'] = df['emojis'].apply(lambda x: ''.join(x))
    # df['emoji_encoded'] = emoji_encoder.transform(df['emojis'])
    # df = df.drop(columns=['emojis'])

    # Align the dataframe with the training columns
    missing_cols = set(train_columns) - set(df.columns)
    print(missing_cols)
    # for col in missing_cols:
    #     df[col] = 0
    df = df[train_columns]

    
    prediction = model.predict(df)
    st.write(f"Estimated Reach: {int(prediction[0])}")
    st.write(f"Estimated Reach based on Benchmark: {((int(prediction[0])/int(benchmark))*100):.2f}%")
    st.write(f"Benchmark: {int(benchmark)}")

# Footer
st.write('---')
st.write('Developed with ❤️ by TeamGary')
