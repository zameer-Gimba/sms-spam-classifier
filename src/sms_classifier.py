## Imports & Configuration
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

tf.random.set_seed(42)
np.random.seed(42)

# Data Loading
train_data = pd.read_csv("data/train-data.tsv", sep="\t", names=["label", "message"])
test_data  = pd.read_csv("data/valid-data.tsv", sep="\t", names=["label", "message"])

## Label Encoding; Convert labels to binary (ham = 0, spam = 1)
label_encoder = LabelEncoder()
train_data["label"] = label_encoder.fit_transform(train_data["label"])
test_data["label"] = label_encoder.transform(test_data["label"])

# Split data into features (X) and labels (y)
X_train, y_train = train_data["message"], train_data["label"]
X_test, y_test = test_data["message"], test_data["label"]

# Convert text data to numerical features
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# Build neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_vectorized.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_vectorized.toarray(), y_train, epochs=10, batch_size=32, validation_data=(X_test_vectorized.toarray(), y_test))

# Define prediction function
def predict_message(message):
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized.toarray())[0][0]
    label = 'spam' if prediction > 0.5 else 'ham'
    return [float(prediction), label]

# Evaluate model
loss, accuracy = model.evaluate(X_test_vectorized.toarray(), y_test)
print(f'Model Accuracy: {accuracy:.4f}')

# Test your function and model.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
