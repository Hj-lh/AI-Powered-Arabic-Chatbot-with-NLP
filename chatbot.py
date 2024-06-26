import tensorflow as tf
import numpy as np
import json
import re
import pickle
import random
from time import sleep
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_alef_ar
from preprocess import load_intents, preprocess_input
nltk.download('punkt')
nltk.download('stopwords')

class Chatbot:
    def __init__(self):
        self.context = {}
        print("Loading model...")
        self.model = tf.keras.models.load_model("models/model.h5")
        print("Model loaded")
        self.data, self.subjects_list = load_intents("data/intents.json")
        with open("data.pickle", "rb") as f:
            self.words, self.labels, _, _ = pickle.load(f)
        print(f"Number of words during prediction: {len(self.words)}")
        print(f"Words: {self.words}")
        print(f"Labels: {self.labels}")

    def bag_of_words(self, s):
        return preprocess_input(s, self.words)

    def extract_entities(self, inp):
        subjects = [subject["subject"] for subject in self.data["subjects"]]
        professors = [professor["professor"] for professor in self.data["professors"]]
        detected_subject = None
        detected_professor = None
        
        for subject in subjects:
            if re.search(subject, inp, re.IGNORECASE):
                detected_subject = subject
                break
        
        for professor in professors:
            if re.search(professor, inp, re.IGNORECASE):
                detected_professor = professor
                break
        
        return detected_subject, detected_professor

    def update_context(self, user_id, tag, subject=None):
        if user_id not in self.context:
            self.context[user_id] = {}
        self.context[user_id]['last_intent'] = tag
        if subject:
            self.context[user_id]['last_subject'] = subject

    def get_last_context(self, user_id):
        if user_id in self.context:
            return self.context[user_id].get('last_intent'), self.context[user_id].get('last_subject')
        return None, None

    def get_response(self, user_id, inp):
        subject, professor = self.extract_entities(inp)
        last_intent, last_subject = self.get_last_context(user_id)

        results = self.model.predict([self.bag_of_words(inp)])[0]
        results_index = np.argmax(results)
        tag = self.labels[results_index]
        confidence = results[results_index]
        print(f'confidence {confidence}, tag {tag}')
        
        if confidence > 0.8:
            if tag == "متابعة" and last_intent:
                tag = last_intent

            for tg in self.data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    
                    if tag == "قائمة_المواد":
                        subjects_taught = self.get_subjects_taught_by(professor)
                        if subjects_taught:
                            subject_list = ", ".join(subjects_taught)
                            responses = [res.replace("{professor}", professor).replace("{subject_list}", subject_list) for res in responses]
                        else:
                            responses = [f"عذراً، لا توجد معلومات عن المواد التي يدرسها {professor} حالياً."]
                    elif tag == "موعد":
                        if subject:
                            self.update_context(user_id, tag, subject)
                            for sub in self.data["subjects"]:
                                if sub["subject"] == subject:
                                    responses = [res.replace("{date}", sub["date"]).replace("{subject}", sub["subject"]) for res in responses]
                                    break
                            else:
                                responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                        else:
                            if last_intent == "موعد" and last_subject:
                                for sub in self.data["subjects"]:
                                    if sub["subject"] == last_subject:
                                        responses = [res.replace("{date}", sub["date"]).replace("{subject}", sub["subject"]) for res in responses]
                                        break
                                else:
                                    responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                            else:
                                responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                    elif tag == "متابعة" and last_intent:
                        responses = [res.replace("{subject}", last_subject if last_subject else "غير معروف") for res in responses]
                    elif tag == "تدريس_الأستاذ":
                        if professor and subject:
                            teaches = self.check_professor_teaches(professor, subject)
                            if teaches:
                                responses = [res.replace("{professor}", professor).replace("{subject}", subject) for res in responses]
                            else:
                                responses = [res.replace("{professor}", professor).replace("{subject}", subject) for res in responses if "لا" in res]
                        else:
                            responses = ["عذراً، لا توجد معلومات كافية لإجابة هذا السؤال."]
                    elif tag == "الأستاذ_للمادة":
                        if subject:
                            professor = self.get_professor_of_subject(subject)
                            responses = [res.replace("{subject}", subject).replace("{professor}", professor) for res in responses]
                        else:
                            responses = ["عذراً، لا توجد معلومات كافية لإجابة هذا السؤال."]
                    self.update_context(user_id, tag, subject)
                    break
            
            sleep(1)
            return random.choice(responses)
        else:
            other_responses = [tg['responses'] for tg in self.data["intents"] if tg['tag'] == "اي شي اخر"][0]
            return random.choice(other_responses)

    def check_professor_teaches(self, professor, subject):
        for prof in self.data["professors"]:
            if prof["professor"] == professor:
                teaches_list = prof["teaches"].split(", ")
                if subject in teaches_list:
                    return True
        return False

    def get_subjects_taught_by(self, professor):
        for prof in self.data["professors"]:
            if prof["professor"] == professor:
                return prof["teaches"].split(", ")
        return []

    def get_professor_of_subject(self, subject):
        for prof in self.data["professors"]:
            teaches_list = prof["teaches"].split(", ")
            if subject in teaches_list:
                return prof["professor"]
        return "غير معروف"
def add_new_subject(self, subject, date):
    new_subject = {"subject": subject, "date": date}
    self.data["subjects"].append(new_subject)
    self.subjects_list.append(subject)
    self.save_data()

def save_data(self):
    with open("data/intents.json", "w", encoding="utf-8") as file:
        json.dump(self.data, file, ensure_ascii=False, indent=4)
