import streamlit as st
import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

# Load NLTK stopwords
stop_words = set(stopwords.words('english'))

# Define text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load the pretrained model
model_filename = r'model/model.pkl'
with open(model_filename, 'rb') as file:
    logistic_model = pickle.load(file)

# Load the pretrained TF-IDF vectorizer
vectorizer_filename = r'model/tfidf_vectorizer.pkl'
with open(vectorizer_filename, 'rb') as file:
    vectorizer = pickle.load(file)

# Define a function to make predictions
def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    transformed_text = vectorizer.transform([preprocessed_text])
    prediction = logistic_model.predict(transformed_text)
    return prediction[0]

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #96C9F4;
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #007bff;
        border-color: #007bff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Web App for Sentiment Analysis')

st.write('This is a web app to classify the sentiment of customer reviews as positive, neutral, or negative.')

# User input
user_input = st.text_area('Enter a customer review:', '')

if st.button('Predict'):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f'The sentiment of the review is: **{prediction}**')
    else:
        st.write('Please enter a review to get a prediction.')

# File upload
st.write('Or upload a file containing customer reviews (CSV format). The file should have a column named "review" or "comment".')

uploaded_file = st.file_uploader('Choose a file')

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Check if the necessary column is present
        if 'review' in df.columns or 'comment' in df.columns:
            column_name = 'review' if 'review' in df.columns else 'comment'
            
            # Preprocess the text and predict sentiment for each review
            df['processed_text'] = df[column_name].apply(preprocess_text)
            transformed_texts = vectorizer.transform(df['processed_text'])
            df['sentiment'] = logistic_model.predict(transformed_texts)
            
            st.write('Sentiment analysis results:')
            st.write(df[[column_name, 'sentiment']])
            
            # Option to download the results
            result_csv = df[[column_name, 'sentiment']].to_csv(index=False)
            st.download_button('Download Results', result_csv, file_name='sentiment_analysis_results.csv', mime='text/csv')
        else:
            st.write('The file must contain a column named "review" or "comment".')
    except Exception as e:
        st.write(f'Error processing the file: {e}')
