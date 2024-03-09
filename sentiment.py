import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
import pickle

# Load the model and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

# Text preprocessing function
def preprocessing(text):
    stopwords_set = set(stopwords.words('english'))
    emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')
    text = re.sub('<[^>]*>', '', text)
    emojis = emoji_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')

    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

# Prediction function
def predict_sentiment(comment):
    preprocessed_comment = preprocessing(comment)
    comment_list = [preprocessed_comment]
    comment_vector = tfidf.transform(comment_list)
    prediction = clf.predict(comment_vector)[0]
    return prediction

# Streamlit UI
def main():
    st.title('Sentiment Analysis')
    # st.subheader('Put Your Comment Here:')
    comment = st.text_area(label='Type your comment here:', height=200)
    if st.button('Predict'):
        if comment.strip() == '':
            st.warning('Please enter some text.')
        else:
            prediction = predict_sentiment(comment)
            if prediction == 1:
                st.write("Positive Comment")
            else:
                st.write("Negative Comment")

if __name__ == '__main__':
    main()
