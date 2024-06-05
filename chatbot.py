import tensorflow as tf
import numpy as np
import random
import json
import re
from time import sleep
import pickle
from preprocess import preprocess_input, load_intents

# Load the model
model = tf.keras.models.load_model("models/model.h5")

# Load data
data, subjects_list = load_intents("data/intents.json")

# Load other necessary data
with open("data.pickle", "rb") as f:
    words, labels, _, _ = pickle.load(f)

print(f"Number of words during prediction: {len(words)}")

def bag_of_words(s, words):
    return preprocess_input(s, words)

def extract_subject(inp):
    subjectNo = re.findall(r'(?<!\d)(\d+)(?!\d)', inp)
    if subjectNo:
        subjectNo = subjectNo[0]
        subject_candidates = [subject for subject in subjects_list if subject in inp]
        if subject_candidates:
            user_subject = subject_candidates[0]
            return f"{user_subject} {subjectNo}"
    return None

def get_response(inp):
    results = model.predict([bag_of_words(inp, words)])[0]
    results_index = np.argmax(results)
    tag = labels[results_index]
    confidence = results[results_index]
    print(f"Predicted Tag: {tag}, Confidence: {confidence}")
    if confidence > 0.8:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                if tag == "موعد":
                    subject = extract_subject(inp)
                    if subject:
                        for sub in data["subjects"]:
                            if sub["subject"] == subject:
                                responses = [res.replace("{date}", sub["date"]).replace("{subject}", sub["subject"]) for res in responses]
                                break
                        else:
                            responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                    else:
                        responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                break
        sleep(1)
        return random.choice(responses)
    else:
        other_responses = [tg['responses'] for tg in data["intents"] if tg['tag'] == "اي شي اخر"][0]
        return random.choice(other_responses)
