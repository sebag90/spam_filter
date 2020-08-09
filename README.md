# Spam filter based on a bag of words and a multinomial naive bayes classifier:

A spam filter written in python. First the input emails are converted into a vector space using a Bag of Words model.  
This is then passed on to a multinomial naive bayes classifier wich will classify new emails and save the results 
in results.json


## Dependencies:
- Natural Language Toolkit: https://www.nltk.org/
- numpy: https://numpy.org/


### Install:
pip3 install -r requirements.txt


### Stopwords:
Source: https://github.com/stopwords-iso/stopwords-iso


## Use:
Simply copy your spam and ham (not spam) email in separate files (ideally one email per file) and put them in the 
input directory. In the same manner, save the emails which you would like to classify in the to_classify directory 
and run filter.py