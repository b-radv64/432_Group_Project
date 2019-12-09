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

# probability threshold

ERROR_THRESHOLD = 0.2
# load our calculated synapse values
synapse_file = 'synapses.json' 
with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])
    
def classify(sentence, show_details=False):
    results = think(sentence, show_details)
    
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s \n classification: %s" % (sentence, return_results))
    return return_results

classify("sudo make me a sandwich")
classify("how are you today?")
classify("talk to you tomorrow")
classify("who are you?")
classify("make me some lunch")
print ()
classify("how was your lunch?", show_details=True)