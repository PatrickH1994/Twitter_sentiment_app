"""
In this file I create a sentiment classifier for tweets by fine-tuning BERT.

Data: sentiment 140 (https://www.tensorflow.org/datasets/catalog/sentiment140)

created by: Patrick Hallila 30/08/2022

"""

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
import tensorflow_text

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import io
import json



DATA_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//training.1600000.processed.noemoticon.csv"
MODEL_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//sentiment_model.h5"

LSTM_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//lstm_sentiment_model.h5"
TOKENIZER_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//Twitter_sentiment//tokenizer.json"

BERT_URL = 'https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/4'
PREPROCESSOR_URL = 'https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3'

TWEET_LENGTH = 200

class data_:
    def fix_target_values(target):
        
        if target == 4:
            return 1
        return target

    def load_data(data_path = DATA_PATH):
        
        #Column names taken from kaggle 
        colnames=['target', 'id', 'date', 'flag', 'user', 'text']

        #Load data
        data = pd.read_csv(DATA_PATH, names = colnames, encoding='latin-1')
        print("Original data shape: {}".format(data.shape))

        #Select columns
        data = data[['target', 'text']]

        #Drop neutral tweets
        data = data[data.target != 2]

        # Clean target variable
        data['target'] = data.target.apply(data_.fix_target_values)
        print("Final data shape: {}".format(data.shape))
        return data

    def create_train_val_test_split(X,y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        return X_train, X_validation, X_test, y_train, y_validation, y_test

class BERT_model:

    def create_BERT_model(X_train, y_train, X_validation, y_validation):
        # Create embedding model
        embedding_model = BERT_model.create_embedding_model()
        print("Embedding model created!\n")

        # create model architecture
        sentiment_model = BERT_model.create_sentiment_model(embedding_model)
        print("Sentiment model created!\n")

        # train model
        history = sentiment_model.fit(X_train, y_train, epochs=10, batch_size=128, validation_data=(X_validation, y_validation))
        print("Training finished!\n")

        #Save model
        sentiment_model.save(MODEL_PATH)
        print("Model saved!\n")

        # Evaluate model
        BERT_model.evaluate_model(sentiment_model, X_test, y_test, history)

        return sentiment_model

    def create_embedding_model(preprocessor_url=PREPROCESSOR_URL, bert_url=BERT_URL):
        
        # load prepocessor for albert
        preprocessor = hub.KerasLayer(preprocessor_url)

        # load albert
        bert_encoder = hub.KerasLayer(bert_url)

        # Freeze the encoder to reduce computation needs
        bert_encoder.trainable = False

        #Instantiates a Keras tensor
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        
        # Prepares the preprocessing layer, i.e. when we go from text to a numeric vector
        encoder_inputs = preprocessor(text_input)
        
        # Prepares the embedding layer
        outputs = bert_encoder(encoder_inputs)
        
        # Takes the [CLS] token, check link below if unclear 
        # https://stackoverflow.com/questions/61331991/bert-pooled-output-is-different-from-first-vector-of-sequence-output
        pooled_output = outputs["pooled_output"]     
        
        # Creates the embedding model
        embedding_model = tf.keras.Model(text_input, pooled_output)
        
        return embedding_model

    def create_sentiment_model(embedding_model):
        model = tf.keras.Sequential()
        #Embedding layer
        model.add(embedding_model)

        # #Layer 1 with dropout and batchnormalization
        # model.add(tf.keras.layers.Dense(128, activation='relu'))
        # model.add(tf.keras.layers.Dropout(.3))
        # model.add(tf.keras.layers.BatchNormalization())

        #Layer 2 with dropout and batchnormalization
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(.3))
        model.add(tf.keras.layers.BatchNormalization())

        #Output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        #Summarise model
        model.summary()

        # Create optimizer 
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)

        # Compile model
        model.compile(optimizer=optimizer,
               loss=tf.keras.losses.BinaryCrossentropy(),
               metrics=['accuracy'])

        return model
    
    def evaluate_model(model, X_test, y_test, history):

        #Print key metrics
        print("Evaluate on test data")
        results = model.evaluate(X_test, y_test)
        print("test loss, test acc:", results)

        # Plot performance
        model.plot_performance(history)

        print("Predictions made by model:")
        for i in [35,78,123]:
            prediction = model.predict(X_test.iloc[i])
            if np.argmax(prediction) == 0:
                value = "Negative"
            else:
                value = "Positive"
            print(f"Message: {X_test[i]} --> {value}, {prediction[np.argmax(prediction)]}\n")

    def plot_performance(history):
        # Plot history: Accuracy
        plt.plot(history.history['accuracy'], label='acc')
        plt.plot(history.history['val_accuracy'], label='val_acc')
        plt.title('Accuracy history')
        plt.ylabel('Accuracy value (%)')
        plt.xlabel('No. epoch')
        plt.show()

class LSTM_model:

    def create_tokenizer(tweets, num_words=6000):
        
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(tweets)

        LSTM_model.save_tokenizer(tokenizer)

        return tokenizer        

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

    def preprocess_data(tweets, tokenizer, max_len=TWEET_LENGTH):
        encoded_docs = tokenizer.texts_to_sequences(tweets)
        padded_sequence = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=200)

        return padded_sequence

    def save_tokenizer(tokenizer, tokenizer_path=TOKENIZER_PATH):
        tokenizer_json = tokenizer.to_json()
        with io.open(tokenizer_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def test_model_accuracy(model, tokenizer, X_test, y_test):
        #Print key metrics
        print("Evaluate on test data")
        X_test = LSTM_model.preprocess_data(X_test, tokenizer)
        results = model.evaluate(X_test, y_test)
        print("test loss, test acc:", results)

def main():

    # load data
    data = data_.load_data()
    data = data.sample(n=1000000)
    print("Data loaded!\n")

    # perform train test split
    X_train, X_validation, X_test, y_train, y_validation, y_test = data_.create_train_val_test_split(data.text, data.target)

    # Create BERT model --> I don't use this because the model performs poorly
    # bert_model = BERT_model.create_BERT_model(X_train, y_train, X_validation, y_validation)

    # Create tokenizer for LSTM model
    tokenizer = LSTM_model.create_tokenizer( X_train)

    # Create LSTM model
    lstm_model = LSTM_model.create_model_architecture()

    # Use padding to make all tweets the same length
    padded_sequence_train = LSTM_model.preprocess_data(X_train, tokenizer)
    padded_sequence_val = LSTM_model.preprocess_data(X_validation, tokenizer)

    # Train the model
    history = lstm_model.fit(padded_sequence_train, y_train, epochs=10, batch_size=64, validation_data=(padded_sequence_val, y_validation))

    # Test accuracy
    LSTM_model.test_model_accuracy(lstm_model, tokenizer, X_test, y_test)

    #Save model
    lstm_model.save_weights(LSTM_PATH)

    

if __name__ == "__main__":
    main()