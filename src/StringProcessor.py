from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.chunk import conlltags2tree, tree2conlltags
import language_check
lctool = language_check.LanguageTool('en-US')
import nltk
nltk.download('punkt', quiet = True)
nltk.download('maxent_ne_chunker', quiet = True)
nltk.download('words', quiet = True)
nltk.download('averaged_perceptron_tagger', quiet = True)

with open('../data/words.txt', 'r') as file:
    data_words = file.read().replace('\n', ' ')
word_dict = data_words.split()
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
def match_similarity(sent2, sent1):
    map1 = {}
    for word in sent1:
        if word not in map1:
            map1[word] = 1
    count = 0
    for word in sent2:
        if word in map1:
            count += 1
    return count
def old_NER(sentence):
    return nltk.ne_chunk(pos_tag(sentence))

#checks grammar, input full sentence string, returns true/false
def grammar_check(sentence):
    matches = lctool.check(sentence)
    return len(matches)==0

#auto-corrects grammar, does not always work
def grammar_auto_correct(sentence):
    return language_check.correct(sentence,lctool.check(sentence))

def full_file_correct(file):
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            print(grammar_auto_correct(line))

def dictionarylookup(word):
    return word in word_dict

#sp = StringProcessor()
if __name__ == "__main__":
    #sentence = "France is in paris is a student at CMU."
    #result = NER(tokenize(sentence))
    #print(result["France"])
    #print(grammar_check('Between 2007 and 2012 ,is Dempsey played for Premier League team Fulham and the club\'s highest Premier League goalscorer of all time?'))
    print(dictionarylookup("Dempsey"))


