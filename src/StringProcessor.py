from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

class StringProcessor:
    # takes in a raw string and outputs a list of tokens
    def tokenize(self, string):
        return word_tokenize(string)
    # takes in a list of tokens and outputs a list of tuples (token, pos)
    def pos_tag(self, listofstrings):
        return nltk.pos_tag(listofstrings)
    # takes in a sentence and outputs NER of the sentence
    def ner(self, sentence):
        return nltk.ne_chunk(self.pos_tag(sentence))

#sp = StringProcessor()
#sentence = "Michael Huang is a student at Carnegie Mellon University."
#print(sp.ner(sp.tokenize(sentence)))



