
from statistics import mode
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.text import tokenizer_from_json
import json

TOKENIZER_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//tokenizer.json"
MODEL_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//lstm_sentiment_model.h5"

TWEET_LENGTH = 200


class model:
    def __init__(self, tweet_length=TWEET_LENGTH):
        self.sentiment_model = model.load_sentiment_model()
        self.tokenizer = model.load_tokenizer()
        self.max_length = tweet_length
    
    def load_sentiment_model(model_path=MODEL_PATH):
        sentiment_model = model.create_model_architecture()
        sentiment_model.load_weights(model_path)
        return sentiment_model

    def load_tokenizer(tokenizer_path = TOKENIZER_PATH):
        with open(tokenizer_path) as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        return tokenizer

    def create_model_architecture(max_length=TWEET_LENGTH, embedding_vector_length = 32, num_words=6000):
        model_architecture = tf.keras.Sequential()
        model_architecture.add(tf.keras.layers.Embedding(num_words, embedding_vector_length, input_length=max_length))
        model_architecture.add(tf.keras.layers.SpatialDropout1D(0.25))
        model_architecture.add(tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5))
        model_architecture.add(tf.keras.layers.Dropout(0.2))
        model_architecture.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model_architecture.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
        print(model_architecture.summary())
        
        return model_architecture

    def preprocess_data(self, tweets):
        encoded_docs = self.tokenizer.texts_to_sequences(tweets)
        padded_sequence = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=self.max_length)

        return padded_sequence

    def predict(self, tweet):
        
        tweet = [tweet]
        tweet = self.preprocess_data(tweet)
        prediction = self.sentiment_model.predict(tweet)
        prediction = prediction.max()
        return prediction

    def evaluate_sentiment(self, tweet):
        prediction = self.predict(tweet)
        if prediction >=0.5:
            sentiment = '**positive**'
        else:
            sentiment = '**negative**'

        score = round(prediction*100, 2)*2 - 100

        message = "Your tweet is " + sentiment + " with a score of: {} on a scale from -100 to 100".format(score)
        return message


def main():
    st.write("# Twitter sentiment analyser")

    """
    This app allows you to post your tweet and then it analyses the sentiment in your tweet.
    If the sentiment is negative you should consider rewriting the tweet before you post it.
    """

    sentiment_model = model()


    "### Input tweet"
    user_input = st.text_input("", "Insert tweet")

    if st.button('Predict'):
        st.write("**Tweet to analyse:**")
        st.write(user_input)
        
        if len(user_input.split(' ')) > 200:
            st.write("Tweet is too long! It needs to be under 200 words.")
        else:
            message = sentiment_model.evaluate_sentiment(user_input)
            st.write("**Result:**")
            st.write(message)
        
        

if __name__=='__main__':
    main()



