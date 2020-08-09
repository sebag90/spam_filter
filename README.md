# Spam filter based on a bag of words and a multinomial naive bayes classifier


# Bag of Words:
A spam filter written in python. First the input emails are converted into a vector space using a Bag of Words model.  
This is then passed on to a multinomial naive bayes classifier wich will classify new emails and save the results 
in results.json


## Dependencies:
- Natural Language Toolkit: https://www.nltk.org/

### Install:
pip3 install -r requirements.txt




### Stopwords:
Source: https://github.com/stopwords-iso/stopwords-iso