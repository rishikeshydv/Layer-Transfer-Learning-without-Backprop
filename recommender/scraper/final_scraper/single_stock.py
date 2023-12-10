# importing packages
import requests
import pytz
from bs4 import BeautifulSoup
import datetime
import csv

#imports for topicSpaceCreator
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from nltk import pos_tag, word_tokenize
import nltk
import pandas as pd


class individual_stock:

    global soup

    r = requests.get('https://www.allrecipes.com/gallery/top-new-recipes-2022/')
    html = r.content 
    soup = BeautifulSoup(html,'html.parser')

    def __init__(self):
        self.p=None

    def topicSpaceCreator(self,scrapedRecipe):
        nltk.download('averaged_perceptron_tagger')
        scrapedRecipeSeries = pd.Series([scrapedRecipe])
        topicSpace = scrapedRecipeSeries.apply(lambda x: str([word for word, pos in pos_tag(word_tokenize(x)) if pos.startswith('N')]))
        return topicSpace

    def recipe(self):
        for w in soup.find_all('p', {'class': 'comp mntl-sc-block mntl-sc-block-html'})[:]:
            model_input = self.topicSpaceCreator(w.text)
                        
        return model_input