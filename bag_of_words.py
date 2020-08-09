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
        self.texts = []
        self.matrix_terms = []
        self.input_number = 0
        self.y = []
        self.__retrieve_texts("ham")
        self.__retrieve_texts("spam")
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
    def __retrieve_texts(self, directory):
        for filename in os.listdir("./input/" + directory):
            path = "./input/" + directory + "/" + filename
            with open(path, "r") as file:
                my_string = file.read()
                cleaned = self.__clean_string(my_string)
                self.texts.append(cleaned)
                if directory == "ham":
                    self.y.append(0)  
                else:
                    self.y.append(1)
        


    # collect stop words after removing whitespaces and /n
    def __collect_stopwords(self):

        with open("stopwords-iso.json") as stopwords:
            data = json.load(stopwords)
        for word in data[self.language]:
            newword = word.replace(" ", "")
            newword2 = newword.replace("\n", "")
            self.stopwords.add(newword2.lower())
        


    # remove stopwords and stem a string. The result is a vector of tokens
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



    # given a list of tokens in a sentence and a list of all tokens in all texts, 
    # this method computes the single sentence vector
    def __calculate_vec(self, sentence_vector):
        string_vec = []
        # iterate over list of all tokens and set vector entry to 0
        for single_term in self.matrix_terms:
            counter = 0
            # iterate over tokens in sentence, if token is present append the number of occurencies 
            for term in sentence_vector:
                if single_term == term:
                    counter = counter + 1
            string_vec.append(counter)
        return string_vec



    # PUBLIC METHODS---------------------------------------------------------------------------------



    def compute_matrix(self):
        self.term_matrix = []
        # iterate over texts, if they are not list (aka not stemmed, stem them)
        for i in range(len(self.texts)):
            if type(self.texts[i]) != list:
                self.texts[i] = self.__str_2_vec(self.texts[i])
                self.__create_matrix_terms(self.matrix_terms, self.texts[i])
        # now calculate the TF vector of each sentence given the complete token list
        for key in self.texts:
            vec = self.__calculate_vec(key)
            self.term_matrix.append(vec)


    # add a sentence to the matrix, this will compute the matrix again
    def add_sentence(self, sentence, label):
        cleaned = self.__clean_string(sentence)
        self.texts[self.input_number] = cleaned
        self.input_number += 1
        self.y.append(label)
        self.compute_matrix()



    def print_matrix(self):
        for i in self.term_matrix:
            print(i)



    def save(self):
        return self.term_matrix, self.y



if __name__ == "__main__":
    bag = BagWords("en")
    bag.compute_matrix()
    bag.print_matrix()
    