#!/usr/bin/env python3
import StringProcessor as sp
import io
import sys
from QuestionScorer import QuestionScorer
import re
import stanfordnlp
import YesNoGenerator
#stanfordnlp.download('en')
import sys
#text_trap = io.StringIO()
#sys.stdout = text_trap
#sys.stderr = text_trap
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
def find_keyword(sentence):
    #nlp = stanfordnlp.Pipeline()
    relation_list = ["nsubj", "obj", "" "nummod", "root", "advmod", "iobj", "amod", ]
    doc = nlp(sentence)
    dependency = doc.sentences[0].dependencies_string()
    parsed_list = []
    for line in dependency.split("\n"):
        # word, index, relation = re.split(r"\".*\"", line)
        line = line.replace("'", "")
        word, index, relation = re.findall(r"\((.*), (.*), (.*)\)", line)[0]
        index = int(index)-1
        parsed_list.append([word, index, relation, []])
    ret = []
    for (i, l) in enumerate(parsed_list):
        w, d, r, c = l[0], l[1], l[2], l[3]
        if len(c) > 0:
            c.append(i)
        if r == "compound":
            parsed_list[d][3].append(i)
    for l in parsed_list:
        w, r, c = l[0], l[2], l[3]
        if len(c) > 0:
            w = ""
            for i in sorted(c):
                w += parsed_list[i][0]
                w += " "
            w = w[:-1]
        if r in relation_list:
            ret.append((w, r))
    return ret

def simplify_sentence(sent):
    leftcomma = -1
    leftparens = -2
    leftparens1 = -2
    new_sent = []
    for i in range(0,len(sent)):
        word = sent[i]
        new_sent.append(word)
        if word == "," and leftcomma < 0:
            leftcomma = i
        elif word == "," and leftcomma > 0:
            new_sent = new_sent[0:leftcomma]
        elif word == "(" and leftparens < 0:
            leftparens = i
        elif word == ")" and leftparens > 0:
            new_sent = new_sent[0:leftparens]
        elif word == "[" and leftparens1 < 0:
            leftparens1 = i
        elif word == "]" and leftparens1 > 0:
            new_sent = new_sent[0:leftparens1]
        elif word == ';':
            return new_sent[0:i]
    return new_sent
def find_first_conjunction(question):
    list_of_conjunctions = [", and", ", but"]
    for conj in list_of_conjunctions:
        idx = question.find(conj)
        if idx > 0:
            return question[0:idx]
    return question

if __name__ == '__main__':

    args = sys.argv
    #numQs = int(args[2])#args[1] # return top numQ questions
    with open('../data/set3/a1.txt', 'r') as file:
        data = file.read().replace('\n', ' ')
    with open("../data/questiondataset.txt", 'r') as f:
        scorer_train_data = f.read().replace('\n', ' ')
    questionscorer = QuestionScorer(scorer_train_data)
    test = QuestionScorer(data)
    numQs = 15
    #print(sys.argv[1])
    #with open(args[1], 'r') as file:
    #    data = file.read().replace('\n', ' ')

    sentences = sp.sent_tokenize(data)
    # POS key: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
    questions = []
    questionidx = 0
    ### Template Matching
    for i in range(0, len(sentences)):
        sent = sentences[i]
        #print(sent)
        sent_tok = simplify_sentence(sp.tokenize(sent))
        #print(sent_tok)
        sent_tok_string = " ".join(sent_tok)
        if sent_tok_string == None or len(sent_tok_string) < 1:
            continue
        importantwords = find_keyword(sent_tok_string)
        #print(importantwords)
        ner_tags = sp.NER(sent_tok)
        sennop = sent_tok_string[0:len(sent_tok_string)-1]
        for (word, pos) in importantwords:
            verb = "blue"
            subj = "jogged"
            if word in ner_tags.keys():
                pattern = " " + word + " "
                if pos == "root":
                    verb = word
                if pos == "nsubj":
                    subj = word
                    if sennop.replace(pattern, " UNK ") != sennop:
                        idx = sennop.find(pattern) + 1
                        if ner_tags[word] == "B-PERSON" or ner_tags[word] == "I-PERSON":
                            questionidx += 1
                            question = sennop.replace(pattern, " who ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                        elif ner_tags[word] == "B-GPE":
                            questionidx += 1
                            question = sennop.replace(pattern, " where ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                        elif not sp.dictionarylookup(word):
                            questionidx += 1
                            question = sennop.replace(pattern, " who ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                        else:
                            questionidx += 1
                            print(word + ": " + pos + ": " + ner_tags[word])
                            question = sennop.replace(pattern, " what ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question)+ "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                if pos == "obj" and verb!= "blue" and subj != "jogged":
                    if sennop.replace(pattern, " UNK ") != sennop:
                        idx = sennop.find(pattern) + 1
                        if ner_tags[word] == "B-PERSON" or ner_tags[word] == "I-PERSON":
                            questionidx += 1
                            question = sennop.replace(pattern, " who ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                        elif ner_tags[word] == "B-GPE":
                            questionidx += 1
                            question = sennop.replace(pattern, " where ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                        elif not sp.dictionarylookup(word):
                            questionidx += 1
                            question = sennop.replace(pattern, " who ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                        else:
                            questionidx += 1
                            print(word + ": " + pos + ": " + ner_tags[word])
                            question = sennop.replace(pattern, " what ")
                            question = question[idx:len(question)]
                            question = find_first_conjunction(question) + "?"
                            if len(question.split()) >= 4:
                                questions.append(question)
                if pos == "amod":
                    # get the word this modifies, need dependency tree thing for that
                    continue

        #Yes/no question maker
        yesno = YesNoGenerator.GenerateYesNo(sent_tok)
        if not (yesno==None):
            questions.append(yesno)
    filteredquestions = []
    for question in questions:
        if sp.grammar_check(question):
            filteredquestions.append(question)

    sys.stdout = sys.__stdout__

    for i in range(0, len(filteredquestions)):
        print(filteredquestions[i % len(filteredquestions)])
    print("unfiltered: \n")
    for i in range(0, len(questions)):
        print(questions[i % len(questions)])
    ### Question Scorer
    #scores = zip(questions, questionscorer.scoreQuestions(questions))
    #print(scores)
    #sortedscores = sorted(scores, key=lambda x: x[1]) # sort by score
    #for i in range(0, numQs):
    #    print(questions[i % len(questions)])