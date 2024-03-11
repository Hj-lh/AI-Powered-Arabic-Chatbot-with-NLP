import tensorflow as tf

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import numpy
import tflearn
import random
import json
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from camel_tools.utils.normalize import normalize_alef_ar 
from time import sleep
import pickle
import re

#--------------------------------DATA--------------------------------
with open("C:/Users/PC/Dev/webDev/myarabicbot/intents.json", encoding="utf-8") as file:
    data = json.load(file)





words = []
labels = []
docs_x = []
docs_y = []


#--------------------------------NLP--------------------------------
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern) #tokenize the pre_written patterns in the json file
        words.extend(wrds)  #add ALL the tokenized patterns in a list
        docs_x.append(wrds)
        docs_y.append(intent["tag"]) #connect the patterns saved in docs_x with their corresponding tag

    if intent["tag"] not in labels: #add the tags into a list
        labels.append(intent["tag"])

global stop_words
stop_words= list(stopwords.words("Arabic")) #save the arabic stop words in a list

    #normalize the words in the lists 
stop_words=[normalize_alef_ar(w) for w in stop_words] #Alef variations to plain a Alef character like (ء)
words=[normalize_alef_ar(w) for w in words] 
                
[words.remove(w) for w in words if w in stop_words] #remove the stop words from the list

st = ISRIStemmer() # Stemmer function 
words = [st.stem(w) for w in words] #apply Stemmer on the words included in the list

words = sorted(list(set(words)))

labels = sorted(labels)


    #--------------------------------MODEL DATA--------------------------------
training = []
output = []

    

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x): #create the bag of words (matrics)
    bag = []

    wrds = [st.stem(w) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


#--------------------------------MODEL--------------------------------
tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #Input layer
#Two hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #output layer (softmax is used to give probability)
net = tflearn.regression(net)

model = tflearn.DNN(net) #train the model


model.fit(training, output, n_epoch=2100, batch_size=8, show_metric=True) #pass the training data
model.save("model.tflearn") #save the model


#--------------------------------CHATBOT--------------------------------
  

def bag_of_words(s, words): #process the user's input
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    
    s_words=[normalize_alef_ar(w) for w in s_words] 
    
    [s_words.remove(w) for w in s_words if w in stop_words]
    
    s_words = [st.stem(w) for w in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return numpy.array(bag)

def extract_subject(inp):
    print(inp)
    subjects = ["هال", "عال"]
    subjectNo = re.findall(r'(?<!\d)(\d+)(?!\d)', inp)
    if subjectNo:
        subjectNo = subjectNo[0]
        subject_candidates = [subject for subject in subjects if subject in inp]
        if subject_candidates:
            user_subject = subject_candidates[0]
            return f"{user_subject} {subjectNo}"
    return None


def get_response(inp): #start the chating process
        
        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        confidence = results[results_index]
        print(f"Predicted Tag: {tag}, Confidence: {confidence}")

        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    if tag == "موعد":
                        subject = extract_subject(inp)
                        
                        if subject is not None:
                            print(subject)
                            for sub in data["subjects"]:
                                if sub["subject"] == subject:
                                    responses = [res.replace("{date}", sub["date"]).replace("{subject}", sub["subject"]) for res in responses]
                                    break
                                else:
                                    responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                        else:
                            print("couldn't prosess the subject")
                            responses = ["عذراً، لا توجد معلومات عن هذه المادة حالياً."]
                        
                    break
                
            sleep(1)
            return random.choice(responses)
            
        else:
            other_responses = [tg['responses'] for tg in data["intents"] if tg['tag'] == "اي شي اخر"][0]
            return random.choice(other_responses)
#this is a test to see if this code will go to the main file in github repository (:D)
#this is another test for branshes in github yooohooo youhouuuu