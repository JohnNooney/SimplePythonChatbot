import nltk
import numpy as np
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# create methods to use in the main code
def tokenize(sentence):
    return nltk.word_tokenize(sentence)  # tokenizes the words in a sentence. ie: each word is element in array

def stem(word):
    return stemmer.stem(word.lower())  # gets the stem of the sentece. ie: only root words

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):  # loop through all words to see if the sentece matches any words, if so update index to 1 from 0
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag
