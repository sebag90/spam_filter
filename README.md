# Spam filter based on a bag of words and a multinomial naive bayes classifier:

A spam filter written in python. First the input emails are converted into a vector space using a Bag of Words model.  
This is then passed on to a multinomial naive bayes classifier wich will classify new emails and save the results 
in results.json


## Dependencies:
- Natural Language Toolkit: https://www.nltk.org/

## Requirements:
* all requirements are saved in envoroment.yml

## Quickstart:
* install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* cd into the root directory of this repository
* create a new environment from the environment.yml file
```
conda env create -f environment.yml
```

* activate the new environment
```
conda activate spam
```


### Stopwords:
Source: https://github.com/stopwords-iso/stopwords-iso


## Use:
Simply copy your spam and ham (not spam) email in separate files (ideally one email per file) and put them in the 
input directory. In the same manner, save the emails which you would like to classify in the to_classify directory 
and run filter.py