import json
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_alef_ar
import pickle
from preprocess import preprocess_input, load_intents

nltk.download('punkt')
nltk.download('stopwords')

# Load intents and subjects data
data, subjects_list = load_intents("data/intents.json")

# Initialize lists
words = []
labels = []
docs_x = []
docs_y = []

# Preprocess data
stop_words = list(stopwords.words("Arabic"))
stop_words = [normalize_alef_ar(w) for w in stop_words]
st = ISRIStemmer()

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        wrds = [normalize_alef_ar(w) for w in wrds if w not in stop_words]
        words.extend([st.stem(w) for w in wrds])
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = sorted(list(set(words)))
labels = sorted(labels)

print(f"Number of words during training: {len(words)}")
print(f"Words: {words}")
print(f"Labels: {labels}")

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = [1 if w in [st.stem(wd) for wd in doc] else 0 for w in words]
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)

# Build and train the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(len(training[0]),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(training, output, epochs=400, batch_size=8, verbose=1)
print("Training finished")

model.save("models/model.h5")
print("Model saved to models/model.h5")
