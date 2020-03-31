#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Kinjal Jain Feb 2020

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse
import math
import copy

from utils import *


class LanguageModel(object):
    """
    Base class for all language models
    """

    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        # write your initialize code below

        self.ngram = ngram
        words = {}
        self.total = 0
        for article in corpus:
            self.total += len(article)
            for word in article:
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
        unkwords = []
        self.counter = 0
        for word in words.keys():
            if words[word] < min_freq:
                unkwords.append(word)
        if 'UNK' not in words:
            words['UNK'] = 0

        for word in unkwords:
            #print("dd")
            words['UNK'] += words[word]
            del words[word]
        #print(words['UNK'])
        for article in corpus:
            for i in range(0, len(article)):
                if article[i] in unkwords:
                    article[i] = 'UNK'
        self.freq = words.copy()
        if uniform:
            for word in self.freq.keys():
                self.freq[word]=1
        if ngram == 1:
            self.model = words
            self.model['END'] = 1
            for word in self.model:
                self.model[word] /= self.total
            if uniform:
                for word in self.model:
                    self.model[word] = 1.0 / len(self.model.keys())
        if ngram == 2:
            self.model = dict()
            for article in corpus:
                for i in range(0, len(article) + 1):
                    if i == 0:
                        if 'START' not in self.model:
                            self.model['START'] = dict()
                        if article[0] not in self.model['START']:
                            self.model['START'][article[0]] = 0
                        self.model['START'][article[0]] += 1
                    else:
                        if article[i - 1] not in self.model:
                            self.model[article[i - 1]] = dict()
                        if i == len(article):
                            if 'END' not in self.model[article[i - 1]]:
                                self.model[article[i - 1]]['END']=0
                            self.model[article[i - 1]]['END'] += 1
                        else:
                            if article[i] not in self.model[article[i - 1]]:
                                self.model[article[i - 1]][article[i]] = 0
                            self.model[article[i - 1]][article[i]] += 1
            self.freqmodel = copy.deepcopy(self.model)
            for word1 in self.model.keys():
                sum = 0.0
                for word2 in self.model[word1].keys():
                    self.counter += 1
                    sum += self.model[word1][word2]
                for word2 in self.model[word1].keys():
                    self.model[word1][word2] /= sum
        if ngram == 3:
            self.model = dict()
            for article in corpus:
                for i in range(0, len(article) + 1):
                    if i == 0:
                        if 'START' not in self.model:
                            self.model['START'] = dict()
                        if 'START' not in self.model['START']:
                            self.model['START']['START'] = dict()
                        if article[0] not in self.model['START']['START']:
                            self.model['START']['START'][article[0]] = 0
                        self.model['START']['START'][article[0]] += 1
                    elif i == 1:
                        if article[0] not in self.model['START']:
                            self.model['START'][article[0]] = dict()
                        if article[1] not in self.model['START'][article[0]]:
                            self.model['START'][article[0]][article[1]] = 0
                        self.model['START'][article[0]][article[1]] += 1
                    else:
                        if article[i - 2] not in self.model:
                            self.model[article[i - 2]] = dict()
                        if article[i - 1] not in self.model[article[i - 2]]:
                            self.model[article[i - 2]][article[i - 1]] = dict()
                        if i == len(article):
                            if 'END' not in self.model[article[i-2]][article[i-1]]:
                                self.model[article[i - 2]][article[i - 1]]['END'] = 0
                            self.model[article[i - 2]][article[i - 1]]['END'] += 1
                        else:
                            if article[i] not in self.model[article[i - 2]][article[i - 1]]:
                                self.model[article[i - 2]][article[i - 1]][article[i]] = 0
                            self.model[article[i - 2]][article[i - 1]][article[i]] += 1
            self.freqmodel = copy.deepcopy(self.model)
            for word1 in self.model.keys():
                for word2 in self.model[word1].keys():
                    sum = 0.0
                    for word3 in self.model[word1][word2].keys():
                        self.counter += 1
                        sum += self.model[word1][word2][word3]
                    for word3 in self.model[word1][word2].keys():
                        self.model[word1][word2][word3] /= sum

    def build(self):
        """
        Build LM from text corpus
        """
        # Write your own implementation here

        pass

    def most_common_words(self, k):
        """
        This function will only be called after the language model has been built
        Your return should be sorted in descending order of frequency
        Sort according to ascending alphabet order when multiple words have same frequency
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        l = []
        if self.ngram == 1:
            for word in self.freq.keys():
                l.append(word)
        elif self.ngram == 2:
            for word1 in self.model.keys():
                for word2 in self.model[word1].keys():
                    l.append(word1+" "+word2)
        elif self.ngram == 3:
            for word1 in self.model.keys():
                for word2 in self.model[word1].keys():
                    for word3 in self.model[word1][word2].keys():
                        l.append(word1+" "+word2+" "+word3)
        l.sort()
        if self.ngram == 1:
            #print(l)
            l = sorted(l, key=(lambda word: 10000 - self.freq[word]))
            #print(l)
            bar = [(x, self.freq[x]) for x in l]
        elif self.ngram == 2:
            l=sorted(l, key=(lambda word: 10000 - self.freqmodel[word.split(' ')[0]][word.split(' ')[1]]))
            bar = [(word, self.freqmodel[word.split(' ')[0]][word.split(' ')[1]]) for word in l]
        elif self.ngram == 3:
            l = sorted(l, key=(lambda word: 10000 - self.freqmodel[word.split(' ')[0]][word.split(' ')[1]][word.split(' ')[2]]))
            bar = [(word, self.freqmodel[word.split(' ')[0]][word.split(' ')[1]][word.split(' ')[2]]) for word in l]
        return bar[0:k]


def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    # Write your own implementation here
    for article in data:
        for j in range(len(article)):
            if article[j] not in models[0].freq:
                article[j] = 'UNK'
    perp = 0.0
    probs = [[],[],[],[]]
    for i in range(0, len(models)):
        prod = 1.0
        total = 0
        if models[i].ngram == 1:
            for article in data:
                total += len(article) + 1
                for word in article:
                    probs[i].append(models[i].model[word])
                    prod -= math.log(models[i].model[word])
                        #print(word + " " + str(models[i].model[word]))

                    #print(prod)
                probs[i].append(models[i].model['END'])
                prod -= math.log(models[i].model['END'])

        elif models[i].ngram == 2:
            for article in data:
                total += len(article) + 1
                for j in range(0,len(article)+1):
                    if j == 0 :
                        if article[j] not in models[i].model['START']:
                            models[i].model['START'][article[j]] = models[i].freq[article[j]] / len(models[i].freq.keys())
                        probs[i].append(models[i].model['START'][article[j]])
                        prod -= math.log(models[i].model['START'][article[j]])
                    elif j == len(article):
                        if 'END' not in models[i].model[article[j-1]]:
                            models[i].model[article[j-1]]['END'] = 1.0 / len(models[i].freq.keys())
                        probs[i].append(models[i].model[article[j-1]]['END'])
                        prod -= math.log(models[i].model[article[j-1]]['END'])

                    else:
                        if article[j] not in models[i].model[article[j - 1]]:
                            models[i].model[article[j-1]][article[j]] = models[i].freq[article[j]] / len(models[i].freq.keys())
                        probs[i].append(models[i].model[article[j - 1]][article[j]])
                        prod -= math.log(models[i].model[article[j - 1]][article[j]])
        else:
            for article in data:
                total += len(article) + 1
                for j in range(0,len(article)+1):
                    if j == 0 :
                        if article[j] not in models[i].model['START']['START']:
                            models[i].model['START']['START'][article[j]] = models[i].freq[article[j]] / len(models[i].freq.keys())
                        probs[i].append(models[i].model['START']['START'][article[j]])
                        prod -= math.log(models[i].model['START']['START'][article[j]])
                    if j == 1 :
                        if article[0] not in models[i].model['START']:
                            models[i].model['START'][article[0]] = dict()
                        if article[1] not in models[i].model['START'][article[0]]:
                            models[i].model['START'][article[0]][article[j]] = models[i].freq[article[j]] / len(models[i].freq.keys())
                        probs[i].append(models[i].model['START'][article[j-1]][article[j]])
                        prod -= math.log(models[i].model['START'][article[j-1]][article[j]])
                    elif j == len(article):
                        if article[j-1] not in models[i].model[article[j-2]]:
                            models[i].model[article[j - 2]][article[j - 1]] = dict()
                        if 'END' not in models[i].model[article[j-2]][article[j-1]]:
                            models[i].model[article[j-2]][article[j-1]]['END'] = 1.0 / len(models[i].freq.keys())
                        probs[i].append(models[i].model[article[j-2]][article[j-1]]['END'])
                        prod -= math.log(models[i].model[article[j-2]][article[j-1]]['END'])

                    else:
                        if article[j-1] not in models[i].model[article[j - 2]]:
                            models[i].model[article[j - 2]][article[j - 1]] = dict()
                        if article[j] not in models[i].model[article[j-2]][article[j-1]]:
                            models[i].model[article[j-2]][article[j-1]][article[j]] = models[i].freq[article[j]] / len(models[i].freq.keys())
                        probs[i].append(models[i].model[article[j-2]][article[j - 1]][article[j]])
                        prod -= math.log(models[i].model[article[j-2]][article[j - 1]][article[j]])

    print(probs)
    for i in range(0,len(probs[2])):
        x = coefs[0]*probs[0][i]+coefs[1]*probs[1][i]+coefs[2]*probs[2][i]+coefs[3]*probs[3][i]
        print(x)
        perp -= math.log(coefs[0]*probs[0][i]+coefs[1]*probs[1][i]+coefs[2]*probs[2][i]+coefs[3]*probs[3][i])
    return math.exp(perp / len(probs[2]))


# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    args = parser.parse_args()
    return args


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    # args = parse_args()
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)

    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)

    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)

    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)

    # calculate perplexity on test file
    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))