#!/usr/bin/python3

import os
import math
import json
from nltk.stem.snowball import SnowballStemmer


class BagWords:

    def __init__(self):
        self.term_matrix = []
        self.stopwords = set()
        self.texts = {}
        self.matrix_terms = set()
        
        self.retrieve_texts()
        self.collect_stopwords("en")
        for key in self.texts:
            self.texts[key] = self.str_2_vec(self.texts[key], "en")
            self.create_matrix_terms(self.matrix_terms, self.texts[key])


    # manual string manipulation
    def clean_string(self, _input_str):
        cln_str = _input_str.replace("[", " ")
        cln_str = cln_str.replace("]", " ")
        cln_str = cln_str.replace("(", " ")
        cln_str = cln_str.replace(")", " ")
        cln_str = cln_str.replace(", ", " ")
        cln_str = cln_str.replace(". ", " ")
        cln_str = cln_str.replace("; ", " ")
        cln_str = cln_str.replace(": ", " ")
        cln_str = cln_str.replace("?", " ")
        cln_str = cln_str.replace("!", " ")
        cln_str = cln_str.replace("""'""", " ")
        cln_str = cln_str.replace('"', " ")
        cln_str = cln_str.replace("""„""", " ")
        cln_str = cln_str.replace("""“""", " ")
        return cln_str


    # retrieve articles
    def retrieve_texts(self):
        self.texts = {}
        for filename in os.listdir('./input'):
            path = "./input/" + filename
            with open(path, "r") as file:
                my_string = file.read()
                cleaned = self.clean_string(my_string)
                self.texts[filename] = cleaned
        

    # collect stop words after removing whitespaces and /n
    def collect_stopwords(self, language):

        with open("stopwords-iso.json") as stopwords:
            data = json.load(stopwords)
        for word in data[language]:
            newword = word.replace(" ", "")
            newword2 = newword.replace("\n", "")
            self.stopwords.add(newword2.lower())



    # remove stopwords and stem a string
    def str_2_vec(self, _input_string, language):
        # extract single words
        splits = _input_string.split()
        cleaned = []
        stemmed = []

        languages = {
            "en" : "english",
            "it" : "italian",
            "de" : "german"
        }

        # if word is not a stop word, save it in a new list (vector)
        for something in splits:
            if something.lower() not in self.stopwords:
                cleaned.append(something)
        # stem
    
        stemmer = SnowballStemmer(languages[language])
        for element in cleaned:
            stemmed.append(stemmer.stem(element))

        return stemmed


    # create matrix term list
    def create_matrix_terms(self, _matrix_terms, _article):
        for term in _article:
            if term not in _matrix_terms:
                _matrix_terms.add(term)
        self.term_matrix = _matrix_terms



    def print_matrix(self):
        for i in self.term_matrix:
            print(i)
    







if __name__ == "__main__":
    bag = BagWords()
    bag.print_matrix()