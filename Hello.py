

from streamlit.logger import get_logger
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import datetime
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')

LOGGER = get_logger(__name__)

# Load the model and necessary transformers
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
# label_encoder = joblib.load('label_encoder.pkl')

def predict_post(data):
    clean_description = ' '.join([word for word in data['description'].split() if word not in (stop)])
    description = vectorizer.transform([clean_description])
    publish_hour = data['publish_hour']
    publish_day = data['publish_day']
    post_type = data['post_type']
    
    # Create a dataframe from the data
    df = pd.DataFrame({
        'Duration (sec)': [data['duration']]
        # 'Publish_hour': [publish_hour],
        # 'Publish_day': [publish_day]
    })
    
    # Add TF-IDF features
    tfidf_df = pd.DataFrame(description.toarray(), columns=vectorizer.get_feature_names_out())
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Add one-hot encoded features for 'Post type' and 'Publish_day'
    for pt in [ 'IG carousel', 'IG image', 'IG reel']:  # Replace with actual post types you have
        df[f'Post type_{pt}'] = 1 if pt == post_type else 0
    for day in ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']:
        df[f'Publish_day_{day}'] = 1 if day == publish_day else 0
    for i in range(24):
        df[f'Publish_hour_{str(i)}'] = 1 if i == publish_hour else 0
    
    prediction = model.predict(df)
    return prediction

st.title("Post Impression Predictor")





# Create input widgets
platforms = ["Instagram", "TikTok", "LinkedIn"]
platform = st.selectbox("Platform", platforms)  

if(platform == "Instagram"):
    post_types = ["IG image", "IG reel", "IG carousel"]
else:
    post_types = ["Image", "Reel", "Carousel"]
post_type = st.selectbox("Post Type", post_types) 
description = st.text_area("Description", "Enter post description here...")
duration = st.number_input("Duration (sec)", 0)
publish_date = st.date_input("Publish Date", datetime.date.today()) 
publish_time = st.time_input("Publish Time", datetime.time(8, 45))  
 
# account_username = st.text_input("Account Username", "user1")


# Predict button
if st.button("Predict"):
    data = {
        'description': description,
        'duration': duration,
        'publish_hour': publish_time.hour,
        'publish_day': publish_date.strftime('%A'),
        'post_type': post_type
    }
    prediction = predict_post(data)
    st.write(f"Impressions: {prediction[0]}")




def run():
    pass
    # st.set_page_config(
    #     page_title="Post predictor ",
    #     page_icon="👋",
    # )

    # st.write("# Welcome to Streamlit! 👋")

    # st.sidebar.success("Select a demo above.")

    # st.markdown(
    #     """
    #     Streamlit is an open-source app framework built specifically for
    #     Machine Learning and Data Science projects.
    #     **👈 Select a demo from the sidebar** to see some examples
    #     of what Streamlit can do!
    #     ### Want to learn more?
    #     - Check out [streamlit.io](https://streamlit.io)
    #     - Jump into our [documentation](https://docs.streamlit.io)
    #     - Ask a question in our [community
    #       forums](https://discuss.streamlit.io)
    #     ### See more complex demos
    #     - Use a neural net to [analyze the Udacity Self-driving Car Image
    #       Dataset](https://github.com/streamlit/demo-self-driving)
    #     - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    # """
    # )



if __name__ == "__main__":
    run()
