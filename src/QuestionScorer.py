import json
from StringProcessor import *
from nltk.lm import Vocabulary
from nltk.lm import Lidstone
import pickle
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
class QuestionScorer:
    def __init__(self, trainstring, n=2):
        self.lm = Lidstone(1,n)
        trainsent = sentence_tokenize(trainstring)
        tokenized = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in trainsent]
        #print(tokenized)
        train_data = [list(nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") )for
                      t in tokenized]
        words = [word for sent in  tokenized for word in sent]
        words.extend(["<s>", "</s>"])
        padded_vocab = Vocabulary(words)
        #print(words)
        self.lm.fit(train_data, padded_vocab)


    def scoreQuestions(self, listofquestions):

        # need to add terminal steps from pos tages to single words
        tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in listofquestions]
        #print(tokenized_text)
        test_data = [list(nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>")) for
                     t in tokenized_text]
        results = []
        #print(test_data)
        for test in test_data:
            #print(self.lm.entropy(test))
            results.append(self.lm.entropy(test))
            #print("MLE Estimates:", [((ngram[-1], ngram[:-1]), self.lm.unmasked_score(ngram[-1], ngram[:-1])) for ngram in test])
        return results

if __name__ == "__main__":
    with open("../data/questiondataset.txt", 'r') as f:
        data = f.read().replace('\n', ' ')
    #qs = QuestionScorer(data)
    #pickle.dump(qs, open("n-gram_scorer.p", "wb"))
    qs = pickle.load(open("../data/n-gram_scorer.p", "rb"))
    print(qs.scoreQuestions(["Why is the sky blue?", "blue blue?", "Why you am alive?", "why you not correct?", "what is latin for lion?", "who killed kennedy?", "Where is the great wall of china?" ]))
#print(test.scoreQuestions(["blah blah blah?", "what effect on other matter allows electromagnetic radiation to be visible?"]))
"""
with open('../data/train-v2.0.json', 'r') as f:
    data_dict = json.load(f)
output = open('questiondataset.txt', 'w')
for something in data_dict['data']:
    para = something['paragraphs']
    for so in para:
        qa = so['qas']
        for qas in qa:
            print(qas['question'])
            output.write(qas["question"] + '\n')
        # print(qa)
"""