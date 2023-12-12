import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('train.csv')

data

data.columns

columns = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

# Loop through the list of columns
for column in columns:
    counts = data[column].value_counts()
    print(column + ':')
    print(counts)
    print()

missing_values = data.isnull().sum()

print("Missing Values Count per Column:")
print(missing_values)

# Preprocess data
data['comment_text'] = data['comment_text'].apply(lambda x: x.lower())  # Lowercase text
data['comment_text'] = data['comment_text'].str.replace('[^\w\s]', '', regex=False)  # Remove punctuation

num_words_in_dataset = data['comment_text'].str.split().explode().nunique()

print(f"Number of unique words in the dataset: {num_words_in_dataset}")

data = data.drop('id', axis=1)

from gensim.models import Word2Vec

# Train the Word2Vec model
corpus = [doc.split() for doc in data['comment_text']]
Word2Vecmodel = Word2Vec(sentences=corpus, vector_size=100, window=10, min_count=3, workers=6)

# Tokenize text data
tokenizer = Tokenizer(num_words=num_words_in_dataset, oov_token='<OOV>')
tokenizer.fit_on_texts(data['comment_text'])

from sklearn.model_selection import train_test_split

# Split set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(train_data['comment_text'])
val_sequences = tokenizer.texts_to_sequences(val_data['comment_text'])
test_sequences = tokenizer.texts_to_sequences(test_data['comment_text'])

# Pad sequences
train_padded = pad_sequences(train_sequences, maxlen=256, truncating='post', padding='post')
val_padded = pad_sequences(val_sequences, maxlen=256, truncating='post', padding='post')
test_padded = pad_sequences(test_sequences, maxlen=256, truncating='post', padding='post')

# Define the vocabulary size and embedding matrix
word_index = tokenizer.word_index
vocab_size = len(word_index) + 1
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in word_index.items():
    if word in Word2Vecmodel.wv.key_to_index:
        embedding_matrix[i] = Word2Vecmodel.wv[word]

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization, Activation

# Define the model
input_layer = Input(shape=(256,))

# Embedding layer
embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=256, trainable=False)(input_layer)

# Bidirectional LSTM layer
lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
lstm_layer = Bidirectional(LSTM(64))(lstm_layer)

output_layers = []
for column in columns:
    dense_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(lstm_layer)
    batch_norm_layer = BatchNormalization()(dense_layer)
    activation_layer = Activation('relu')(batch_norm_layer)
    dropout_layer = Dropout(0.2)(activation_layer)
    output = Dense(6, activation='sigmoid')(dropout_layer)
    output_layers.append(output)

# Create the model
model = Model(inputs=input_layer, outputs=output_layers)

from tensorflow.keras.optimizers import Adam
# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display the model summary
model.summary()

columns = ['toxic', 'severe_toxic', 'obscene', 'threat',
       'insult', 'identity_hate']

# Train the model
history = model.fit(train_padded, [train_data[columns] for column in columns], epochs=15, batch_size=32, validation_data=(val_padded, [val_data[columns] for column in columns]))

# Evaluate the model on the test set
results = model.evaluate(test_padded, [test_data[column] for column in columns])

# Display the evaluation results
for i, metric in enumerate(model.metrics_names):
    print(f"{metric}: {results[i]}")

model.save('Comment_toxicity.keras')

text = "you fucking fool"
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=256, padding='post', truncating='post')
predictions = model.predict(padded_sequence)


# Display the predictions with confidence levels
for i, column in enumerate(columns):
    predicted_probabilities = predictions[i][0]
    target_class_index = i
    confidence_level = f"{predicted_probabilities[target_class_index] * 100:.2f}%"
    prediction_label = "Positive" if predicted_probabilities[target_class_index] > 0.5 else "Negative"
    print(f"{column}: Confidence Level = {confidence_level}, Prediction = {prediction_label}")


binary_predictions = [[1 if prob[i] > 0.5 else 0 for i in range(len(columns))] for prob in predictions[0]]
print("Binary Predictions:", binary_predictions)

text = "I like talking about things that make me happy"
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=256, padding='post', truncating='post')
predictions = model.predict(padded_sequence)


# Display the predictions with confidence levels
for i, column in enumerate(columns):
    predicted_probabilities = predictions[i][0]
    target_class_index = i
    confidence_level = f"{predicted_probabilities[target_class_index] * 100:.2f}%"
    prediction_label = "Positive" if predicted_probabilities[target_class_index] > 0.5 else "Negative"
    print(f"{column}: Confidence Level = {confidence_level}, Prediction = {prediction_label}")


binary_predictions = [[1 if prob[i] > 0.5 else 0 for i in range(len(columns))] for prob in predictions[0]]
print("Binary Predictions:", binary_predictions)

text = "I will hurt and kill all your family members you worthless piece of shit"
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=256, padding='post', truncating='post')
predictions = model.predict(padded_sequence)


# Display the predictions with confidence levels
for i, column in enumerate(columns):
    predicted_probabilities = predictions[i][0]
    target_class_index = i
    confidence_level = f"{predicted_probabilities[target_class_index] * 100:.2f}%"
    prediction_label = "Positive" if predicted_probabilities[target_class_index] > 0.5 else "Negative"
    print(f"{column}: Confidence Level = {confidence_level}, Prediction = {prediction_label}")


binary_predictions = [[1 if prob[i] > 0.5 else 0 for i in range(len(columns))] for prob in predictions[0]]
print("Binary Predictions:", binary_predictions)

text = "I hate you and your black ass, get the fuck out of here"
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=256, padding='post', truncating='post')
predictions = model.predict(padded_sequence)


# Display the predictions with confidence levels
for i, column in enumerate(columns):
    predicted_probabilities = predictions[i][0]
    target_class_index = i
    confidence_level = f"{predicted_probabilities[target_class_index] * 100:.2f}%"
    prediction_label = "Positive" if predicted_probabilities[target_class_index] > 0.5 else "Negative"
    print(f"{column}: Confidence Level = {confidence_level}, Prediction = {prediction_label}")


binary_predictions = [[1 if prob[i] > 0.5 else 0 for i in range(len(columns))] for prob in predictions[0]]
print("Binary Predictions:", binary_predictions)

text = "I will find you and make you pay dearly for this, I suggest you run"
sequence = tokenizer.texts_to_sequences([text])
padded_sequence = pad_sequences(sequence, maxlen=256, padding='post', truncating='post')
predictions = model.predict(padded_sequence)


# Display the predictions with confidence levels
for i, column in enumerate(columns):
    predicted_probabilities = predictions[i][0]
    target_class_index = i
    confidence_level = f"{predicted_probabilities[target_class_index] * 100:.2f}%"
    prediction_label = "Positive" if predicted_probabilities[target_class_index] > 0.5 else "Negative"
    print(f"{column}: Confidence Level = {confidence_level}, Prediction = {prediction_label}")


binary_predictions = [[1 if prob[i] > 0.5 else 0 for i in range(len(columns))] for prob in predictions[0]]
print("Binary Predictions:", binary_predictions)