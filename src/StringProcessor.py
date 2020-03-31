from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import sent_tokenize
# takes in a raw string and outputs a list of tokens
def tokenize(self, string):
    return word_tokenize(string)
# takes in a list of tokens and outputs a list of tuples (token, pos)
def pos_tag(self, listofstrings):
    return nltk.pos_tag(listofstrings)
# takes in a aword and outputs what NER BERT classifies it as
def BERT_NER(self, word):
    #TODO install and add BERT pretrained model
    return "BLOC"
def sentence_tokenize(string):
    return sent_tokenize(string)