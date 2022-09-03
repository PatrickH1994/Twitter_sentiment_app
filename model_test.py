import tensorflow as tf
from tensorflow import keras
from keras_preprocessing.text import tokenizer_from_json
import tensorflow_hub as hub
import json
import numpy as np

TOKENIZER_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//tokenizer.json"
MODEL_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//lstm_sentiment_model.h5"

TWEET_LENGTH = 200

def create_model_architecture(embedding_vector_length = 32, num_words=6000, maxlen=TWEET_LENGTH):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(num_words, embedding_vector_length, input_length=maxlen))
    model.add(tf.keras.layers.SpatialDropout1D(0.25))
    model.add(tf.keras.layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model

def load_model(model_path = MODEL_PATH):
    model = create_model_architecture()
    model.load_weights(model_path)
    return model

def preprocess_data(tweets, tokenizer, max_len=TWEET_LENGTH):
    encoded_docs = tokenizer.texts_to_sequences(tweets)
    padded_sequence = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=200)

    return padded_sequence

def main():

    # Load model
    sentiment_model = load_model()

    # Load tokenizer
    with open(TOKENIZER_PATH) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    # Show model architecture
    sentiment_model.summary()

    # Predict model
    twitter_message = ["Shamima Begum, who fled the UK aged 15 to join IS, was killed in Syria by an intelligence agent for Canada",
    "Shamima Begum, who fled the UK aged 15 to join IS, was helped in Syria by an intelligence agent for Canada"]
    twitter_message = preprocess_data(twitter_message, tokenizer)
    print(sentiment_model.predict(twitter_message))
    print("The example shows that the model is able to understand that being killed is negative, whereas being helped is positive.")

if __name__=="__main__":
    main()