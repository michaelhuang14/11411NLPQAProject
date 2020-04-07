from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
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
    list = tree2conlltags(nltk.ne_chunk(pos_tag(sentence)))
    dict = {}
    for (word, pos, ner) in list:
        dict[word] = ner
   # for (word)
    return dict
def sentence_tokenize(string):
    return sent_tokenize(string)
def NERtree(sentence): #needed?
    return nltk.ne_chunk(pos_tag(sentence))
def match_similarity(sent1, sent2):
    map1 = {}
    for word in sent1:
        if word not in map1:
            map1[word] = 1
    count = 0
    for word in sent2:
        if word in map1:
            count += 1
    return count

#sp = StringProcessor()
if __name__ == "__main__":
    sentence = "France is in paris is a student at CMU."
    result = NER(tokenize(sentence))
    print(result["France"])



