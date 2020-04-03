from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# takes in a raw string and outputs a list of tokens
def tokenize(string):
    return word_tokenize(string)
# takes in a list of tokens and outputs a list of tuples (token, pos)
def pos_tag(listofstrings):
    return nltk.pos_tag(listofstrings)
# takes in a sentence and outputs NER of the sentence
def NER(sentence):
    return nltk.ne_chunk(pos_tag(sentence))
def sentence_tokenize(string):
    return sent_tokenize(string)

#sp = StringProcessor()
if __name__ == "__main__":
    sentence = "Michael Huang is a student at Carnegie Mellon University."
    result = NER(tokenize(sentence))
    for chunk in result:
        print(chunk)



