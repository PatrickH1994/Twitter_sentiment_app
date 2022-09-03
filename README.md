# Twitter_sentiment_app

Polarization on social media platforms is an increasingly pressing issue. One potential reason why people publish polarizing content may be that they do it in the spur of the moment and thus do not think about what they are publishing.

This project aims to adress that issue by creating an app that measures the sentiment of a tweet before it is sent. The project consists of the following components:

1) Sentiment model - I used tensorflow to create a sentiment classification model using a LSTM neural network and Sentiment140 as the dataset. My original intention was to use BERT but I didn't have access to a GPU to train it and LSTM models achieve a similar level of performance whilst being computationally more efficient.
2) Test file - I tested that the model gave reasonable results in the test file.
3) APP - I created the app using streamlite. In the app the user can input their tweet and the app calculates the sentiment of the tweet.

Some issues with the projects:

- I use a tokenizer to prepare the data but the file size of the tokenizer is too big to upload on GitHub.
- The app needs to be refreshed after every input (currently working on this)
- Current model accuracy could be improved from 82%