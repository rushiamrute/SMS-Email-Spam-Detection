import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
import nltk
nltk.download('punkt_tab')

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open("model.pkl",'rb'))
def convert(text):
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

st.title("Email/SMS spam Classifier")

input_sms = st.text_area("Enter the message ")

if st.button('Predict'):

    transformed_sms = convert(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
# preprocess

    
