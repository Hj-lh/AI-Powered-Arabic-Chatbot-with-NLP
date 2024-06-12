import json
import nltk
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_alef_ar
import numpy as np
nltk.download('punkt')
nltk.download('stopwords')
def load_intents(filepath):
    with open(filepath, encoding="utf-8") as file:
        data = json.load(file)
    subjects_list = [subject["subject"].split()[0] for subject in data["subjects"]]
    return data, subjects_list

def preprocess_input(s, words):
    stop_words = list(stopwords.words("Arabic"))
    stop_words = [normalize_alef_ar(w) for w in stop_words]
    st = ISRIStemmer()#giving the root

    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [normalize_alef_ar(w) for w in s_words if w not in stop_words]
    s_words = [st.stem(w) for w in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag).reshape(-1, len(words))
