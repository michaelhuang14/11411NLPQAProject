#!/usr/bin/env python3
import StringProcessor as sp
from QuestionScorer import QuestionScorer
import re
import stanfordnlp
#stanfordnlp.download('en')
import sys

def find_keyword(sentence):
    relation_list = ["nsubj", "obj", "" "nummod", "root", "compound", "advmod", "iobj", "amod", ]

    nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
    doc = nlp(sentence)

    dependency = doc.sentences[0].dependencies_string()
    ret = []
    for line in dependency.split("\n"):
        # word, index, relation = re.split(r"\".*\"", line)
        line = line.replace("'", "")
        word, index, relation = re.findall(r"\((.*), (.*), (.*)\)", line)[0]
        if relation in relation_list:
            ret.append((word, relation))
    return ret

if __name__ == '__main__':
    #nquestionscorer = QuestionScorer()
    args = sys.argv
    numQs = int(args[2])#args[1] # return top numQ questions
    #with open('../data/data.txt', 'r') as file:
    #    data = file.read().replace('\n', ' ')
    #print(sys.argv[1])
    with open(args[1], 'r') as file:
        data = file.read().replace('\n', ' ')

    sentences = sp.sent_tokenize(data)
    # is there a way to trim all extra clauses from a sentence? for exapmle, remove things in between two commas,
    # remove things after semi-colons, etc.
    # POS key: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
    questions = []
    ### Template Matching
    for i in range(0, len(sentences)):
        importantwords = find_keyword(sentences[i])
        print(importantwords)
        ner_tags = sp.NER(sp.tokenize(sentences[i]))
        sennop = sentences[i][0:len(sentences[i])-1]
        for (word, pos) in importantwords:
            if pos == "nsubj":
                if ner_tags[word] == "B-PERSON":
                    questions.append(sennop.replace(word, "Who")[0:len()])
                elif ner_tags[word] == "B-GPE":
                    questions.append(sennop.replace(word, "Where") + "?")
                else:
                    questions.append(sennop.replace(word, "what") + "?")
        """if word == 'is' and i > 0 and i < len(POS)-1: # Template for X is Y
            # TODO instead of just using adjacent words, we can try using dependent words from a dependency tree
            # maybe we can use a wordnet to exchange X and Y for synonyms
            (X, X_pos) = POS[i-1]
            (Y, Y_pos) = POS[i+1]
            if Y_pos == 'JJ': # adjective
                questions.append('What kind of ' + X + "?")
            elif Y_pos == 'NN' or Y_pos == "NNS": # noun
                questions.append('What is/are ' + X + "(s)?")
            else:
                if X_pos == "NNP": # proper noun
                    X_NER = stringprocessor.BERT_NER(X)
                    if X_NER == 'LOC':
                        questions.append("Where is " + X + "?")
                    elif X_NER == 'PNAME':
                        questions.append("Who is " + X + "?")
                    else:
                        # add the questions anyway, if they are bad the scorer should filter them out
                        questions.append("What is " + X + "?")
                        questions.append("What is " + Y + "?")
        ## TODO add more templates here, create template class
"""
    for i in range(0, numQs):
        if i < len(questions):
            print(questions[i])
        else:
            print("no more questions")
    ### Question Scorer
    """ scores = questionscorer.scoreQuestions(questions)
    sortedscores = sorted(scores, key=lambda x: x[1]) # sort by score
    for i in range(0, numQs):
        if i >= len(sortedscores):
            print("no more questions")
            break
        (Q) = sortedscores[i]
        print(Q+ '\n')"""