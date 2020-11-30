#!/usr/bin/python3

import os
import re
import math
import json
import string
import numpy as np
from nltk.stem.snowball import SnowballStemmer


class BagWords:

    def __init__(self, language):
        self.language = language
        self.vocabulary = []
        self.matrix = []
        self.texts = []

        with open("stopwords-iso.json") as stopwords:
            stopwords = json.load(stopwords)

        self.stopwords = set(stopwords[language])

    
    def clean_string(self, input_str):
        """
        remove punctuation
        """
        punctuation = [i for i in string.punctuation] + ['„', '“', '”', '–']
        
        for char in punctuation:
            input_str = input_str.replace(char, " ")

        return input_str


    def str_2_vec(self, input_string):
        """
        remove stopwords
        stem the sentence
        return a vector (list) of tokens
        """
        # extract single words
        splits = input_string.split()
        cleaned = []
        stemmed = []

        languages = {
            "en" : "english",
            "it" : "italian",
            "de" : "german"
        }

        # if word is not a stop word, save it in a new list (vector)
        for word in splits:
            if word.lower() not in self.stopwords:
                cleaned.append(word)
        
        # stem
        stemmer = SnowballStemmer(languages[self.language])
        for element in cleaned:
            stemmed.append(stemmer.stem(element))

        return stemmed


    # create matrix term list
    def create_vocabulary(self, sentence):
        """
        takes as input a sentence in form of a stemmed vector
        adds new tokens to the vocabulary
        """
        for term in sentence:
            if term not in self.vocabulary:
                self.vocabulary = np.append(self.vocabulary, term)


    def calculate_vec(self, sentence_vector):
        """
        given a sentence in the form of a list of stemmed tokens
        this method calculates a term frequency vector based on 
        the vocabulary.

        Input: vector of tokens
        Output: term frequency vector
        """

        self.vocabulary = np.array(self.vocabulary) 
        sentence_vector = np.array(sentence_vector)

        indexes = [np.where(x == self.vocabulary)[0][0] for x in sentence_vector]
        result = np.zeros(self.vocabulary.shape)

        for i in indexes:
            result[i] += 1
        
        return result



    # PUBLIC METHODS---------------------------------------------------------------------------------



    def compute_matrix(self):
        """
        given all the sentences added, the bag of words will compute
        the term frequency matrix
        """
        self.matrix = []
        # add all sentences to vocabulary
        for i in range(len(self.texts)):
            if not isinstance(self.texts[i], list):
                self.texts[i] = self.str_2_vec(self.texts[i])
                self.create_vocabulary(self.texts[i])
        
        # calculate matrix
        for line in self.texts:
            vec = self.calculate_vec(line)
            self.matrix.append(vec)


    def add_sentence(self, sentence):
        """
        adds a sentence to the matrix
        """
        cleaned = self.clean_string(sentence)
        self.texts.append(cleaned)


    # TODO: function for TF-IDF
        # ML - ue2

    # TODO: function to calculate sentence similarity
        # add sentence
        # calculate matrix
        # calculate every dot product, report highest

if __name__ == "__main__":
    bag = BagWords("en")
    
    texts = [
        "I was going to the shop and saw a new car",
        "my new car is ideal for going to the shop",
        "my computer is less expensive than my car",
        "my car is more expensive than my computer"
    ]
    for text in texts:
        bag.add_sentence(text)

    bag.compute_matrix()
    for i in bag.matrix:
        print(i)