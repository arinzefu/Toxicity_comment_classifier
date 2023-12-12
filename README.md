# Toxicity_comment_classifier
 Tensorflow model
This dataset originates from the Jigsaw Comment Toxicity Challenge on Kaggle, designed for multi-label binary classification to identify toxicity in text. The output layers correspond to various toxicity labels such as 'toxic,' 'severe_toxic,' 'obscene,' 'threat,' 'insult,' and 'identity_hate.' The dataset includes label counts for each category, with values like toxic: 0 (non-toxic) - 144277, 1 (toxic) - 15294.

The Python script carries out text data preprocessing on the 'comment_text' column, converting it to lowercase, removing punctuation, and training a Word2Vec model. It utilizes the Tokenizer class from Keras for tokenization, followed by dataset splitting, text data conversion to sequences, and padding sequences to a maximum length of 256. The script defines a neural network model using Keras, incorporating an embedding layer, bidirectional LSTM layers, and multiple output layers for binary classification.

Post-training, each column is individually tested on the test set, and the script displays the accuracy for each toxicity category. However, the model faces challenges in accurately identifying 'identity_hate,' as indicated during evaluation on custom input texts, where it exhibits lower confidence levels for this specific category.
