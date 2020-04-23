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
import spacy
import neuralcoref
# Installation of spacy and neuralcoref
# pip install spacy==2.1.0 
# pip install neuralcoref --no-binary neuralcoref
# python -m spacy download en

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
    counts = 0
    for match in matches:
        if match.locqualityissuetype not in ['misspelling']:
            counts += 1
    return counts==0

#auto-corrects grammar, does not always work
def grammar_auto_correct(sentence):
    return language_check.correct(sentence,lctool.check(sentence))

def full_file_correct(file):
    with open(file,'r') as f:
        lines = f.readlines()
        for line in lines:
            print(grammar_auto_correct(line))

def coreference(string):
    coref_pipe = spacy.load("en")
    neuralcoref.add_to_pipe(coref_pipe)
    doc = coref_pipe(string)
    return doc._.coref_resolved

def coreference_resolve_doc(doc):
    sentences = sent_tokenize(doc)
    startidx = 3
    for idx in range(startidx,len(sentences)):
        dereferenced = coreference(" ".join(sentences[idx-startidx:idx]))
        sentences[idx-startidx:idx] = sent_tokenize(dereferenced)
    return " ".join(sentences)

def dictionarylookup(word):
    return word in word_dict

def rank_sentences(sentencelist):
    tokenizedlist = [tokenize(x) for x in sentencelist]
    scores = [len(x) for x in tokenizedlist]
    for i in range(0,len(sentencelist)):
        if ',' in tokenizedlist[i]:
            scores[i] += 80
        if 'and' in tokenizedlist[i]:
            scores[i] += 40
        if sentencelist[i][1:].islower():
            scores[i] += 60
        if tokenizedlist[i][0]=='On' or tokenizedlist[i][0]=='In':
            scores[i] += 100
        if not tokenizedlist[i][-1]=='.':
            scores[i] += 200
        if '\n' in sentencelist[i] or '...' in sentencelist[i]:
            scores[i] += 200
        if scores[i] < 6:
            scores[i] += 20
    scores = zip(sentencelist, scores)
    res = sorted(scores, key=lambda x: x[1])
    return [x for (x,_) in res]



#sp = StringProcessor()
if __name__ == "__main__":
    #sentence = "France is in paris is a student at CMU."
    #result = NER(tokenize(sentence))
    #print(result["France"])
    #print(grammar_check('Where winning an award , as well as Dempsey receiving an award for his goal ?'))
    #print(coreference("Michael has three homeworks due. He is very sad. John is done with his homework. He is happy."))
    #with open("../data/set1/a1.txt",'r',encoding='utf8') as f:
        #a = f.read()
        #print(rank_sentences(sent_tokenize(a)))
    #print("did Leo eat yet?".capitalize())
    q = "did Leo eat yet?"
    q = q[0].upper()+q[1:]
    print(q)





