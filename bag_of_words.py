#!/usr/bin/python3

import os
import math
import json
from nltk.stem.snowball import SnowballStemmer


class BagWords:


    def __init__(self, language):
        self.language = language
        self.term_matrix = []
        self.stopwords = set()
        self.texts = {}
        self.matrix_terms = []
        
        self.__retrieve_texts()
        self.__collect_stopwords()
        


    # manual string manipulation
    def __clean_string(self, _input_str):
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
    def __retrieve_texts(self):
        self.texts = {}
        for filename in os.listdir('./input'):
            path = "./input/" + filename
            with open(path, "r") as file:
                my_string = file.read()
                cleaned = self.__clean_string(my_string)
                self.texts[filename] = cleaned
        

    # collect stop words after removing whitespaces and /n
    def __collect_stopwords(self):

        with open("stopwords-iso.json") as stopwords:
            data = json.load(stopwords)
        for word in data[self.language]:
            newword = word.replace(" ", "")
            newword2 = newword.replace("\n", "")
            self.stopwords.add(newword2.lower())
        

    # remove stopwords and stem a string
    def __str_2_vec(self, _input_string):
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
    
        stemmer = SnowballStemmer(languages[self.language])
        for element in cleaned:
            stemmed.append(stemmer.stem(element))

        return stemmed


    # create matrix term list
    def __create_matrix_terms(self, _matrix_terms, _article):
        for term in _article:
            if term not in _matrix_terms:
                _matrix_terms.append(term)
        self.matrix_terms = _matrix_terms


    def __calculate_vec(self, key):
        string_vec = []
        for single_term in self.matrix_terms:
            counter = 0
            for term in self.texts[key]:
                if single_term == term:
                    counter = counter + 1
            string_vec.append(counter)
        return string_vec


    # PUBLIC METHODS---------------------------------------------------------------------------------


    def compute_matrix(self):
        for key in self.texts:
            self.texts[key] = self.__str_2_vec(self.texts[key])
            self.__create_matrix_terms(self.matrix_terms, self.texts[key])
        for key in self.texts:
            vec = self.__calculate_vec(key)
            self.term_matrix.append(vec)

        return self.term_matrix

    def print_matrix(self):
        for i in self.term_matrix:
            print(i)
        
    



if __name__ == "__main__":
    bag = BagWords("en")
    a = bag.compute_matrix()
    bag.print_matrix()
    