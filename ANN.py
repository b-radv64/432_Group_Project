# use natural language toolkit
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import os
import json
import datetime
import numpy as np
import time
import json
import numpy as np
import time
stemmer = LancasterStemmer()

training_data = []
#training_data = [
#    {"class": "security", "sentence": "A session that has been inactive for more than ten minutes is terminated "},
#    {"class": "security", "sentence": "- Disable accounts after 90 days of inactivity"},
#    {"class": "security", "sentence": "- Passwords can be changed by the associated user only once in a 2-day period"},
#    {"class": "security", "sentence": "- Security decisions are not made based on user-supplied file names and paths."},
#    {"class": "privacy",
#     "sentence": "(1) An authorization or other express legal permission from an individual to use or disclose protected health information for the research;"},
#    {"class": "privacy",
#     "sentence": "(1) An individual has a right to receive an accounting of disclosures of protected health information made by a covered entity in the six years prior to the date on which the accounting is requested, except for disclosures:"},
#    {"class": "privacy",
#     "sentence": "(1) Not use or further disclose the information other than as permitted by the data use agreement or as otherwise required by law;"},
#    {"class": "privacy",
#     "sentence": "(1) To conduct an evaluation relating to medical surveillance of the workplace; or"}
#]
#print(training_data)
#pattern = json.loads(training_data_test)
# security data
# privacy data
words = []
classes = []
documents = []
ignore_words = ['.', ';', ',', '-', ':', '\n', '\'']
for pattern in training_data:
    w = nltk.word_tokenize(pattern['sentence'])  # tokenizes words in the sentence
    words.extend(w)  # add to a word list
    documents.append((w, pattern['class']))  # add to documents
    if pattern['class'] not in classes:  # add to classes list (privacy and security)
        classes.append(pattern['class'])
words = [stemmer.stem(w.lower()) for w in words if
         w not in ignore_words]  # stems(playing --> play) and lowercases each word(SONG --> song)
words = list(set(words))
classes = list(set(classes))  # removes dupes
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique stem words", words)
training = []
output = []
output_empty = [0] * len(classes)  # empty array for outputs
for doc in documents:  # training set bag of words
    bag = []  # initializes the bag array
    pattern_words = doc[0]  # list of tokenized words
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]  # stems and lowercases each word
    for w in words:  # fills array up with bag of words
        bag.append(1) if w in pattern_words else bag.append(0)
    training.append(bag)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)
i = 0
w = documents[i][0]
print([stemmer.stem(word.lower()) for word in w])
print(training[i])
print(output[i])
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
                    
    return(np.array(bag))
    
def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentence:", sentence, "\n bow:", x)
    # input layer is our bag of words
    l0 = x
    # matrix multiplication of input and hidden layer
    l1 = sigmoid(np.dot(l0, synapse_0))
    # output layer
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

# ANN and Gradient Descent code from https://iamtrask.github.io//2015/07/27/python-network-part2/
def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
    
    print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else '') )
    print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X),len(X[0]),1, len(classes)) )
    np.random.seed(1)
    
    last_mean_error = 1
    # randomly initialize our weights with mean 0
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1
    
    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)
    
    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
     
    for j in iter(range(epochs+1)):
        
        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))
            
        layer_2 = sigmoid(np.dot(layer_1, synapse_1))
        
        # how much did we miss the target value?
        layer_2_error = y - layer_2
        
        if (j% 10000) == 0 and j > 5000:
            # if this 10k iteration's error is greater than the last iteration, break out
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("delta after "+str(j)+" iterations:" + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
            
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)
        
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
            
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update
        
    now = datetime.datetime.now()
    
    # persist synapses
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    synapse_file = "synapses.json"
    
    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("saved synapses to:", synapse_file)
    
    
    
start_time = time.time()

train(X = np.array(training), y = np.array(output), hidden_neurons=20, alpha=0.1, epochs=100000, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("processing time:", elapsed_time, "seconds")
