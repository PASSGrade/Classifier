import streamlit as st
import pickle
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidfSms = pickle.load(open('vectorizer.pkl', 'rb'))
modelSms = pickle.load(open('model.pkl', 'rb'))

tfidfEmail = pickle.load(open('vectorizeremail.pkl', 'rb'))
modelEmail = pickle.load(open('modelemail.pkl', 'rb'))

st.title("SMS / Email Spam Classifier")

option = st.selectbox(
    'What would you like test?',
    ('SMS', 'Email'))
st.write('You selected:', option)


def predict_result(tfidf, model):
    if st.button('Predict'):
        # 1. preprocess
        transform_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transform_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("spam")
        else:
            st.header("Not Spam")


if option == 'SMS':
    input_sms = st.text_area("Enter the SMS")
    predict_result(tfidfSms, modelSms)
else:
    input_sms = st.text_area("Enter the Email")
    predict_result(tfidfEmail, modelEmail)
