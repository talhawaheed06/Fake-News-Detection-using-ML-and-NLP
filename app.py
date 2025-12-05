# Uncomment the line above if you want to save this file in Colab to download it

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Load Resources ---
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --- Preprocessing ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- UI ---
st.title("🕵️‍♂️ Fake News Detector")
news_text = st.text_area("News Content", height=200)

if st.button("Check Veracity"):
    if news_text.strip():
        processed_text = preprocess_text(news_text)
        vector_input = vectorizer.transform([processed_text])
        prediction = model.predict(vector_input)[0]
        
        if prediction == 1: 
            st.error("🚨 RESULT: FAKE NEWS DETECTED")
        else:
            st.success("✅ RESULT: REAL NEWS")
    else:
        st.warning("Please enter some text.")
