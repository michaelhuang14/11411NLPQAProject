from nltk.corpus import treebank
from nltk import treetransforms
from nltk import induce_pcfg
from nltk.parse import pchart
import nltk
from nltk import Nonterminal
from nltk.lm import MLE
from StringProcessor import *
from nltk.lm import Vocabulary
from nltk.lm import Lidstone
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
class QuestionScorer:
    # TODO should use PCFG to generate score for each quesiton and output a list of tuples (question, score)
    def __init__(self, trainstring, n=2):
        self.lm = Lidstone(1,n)
        trainsent = sentence_tokenize(trainstring)
        tokenized = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in trainsent]
        print(tokenized)
        train_data = [list(nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") )for
                      t in tokenized]
        words = [word for sent in  tokenized for word in sent]
        words.extend(["<s>", "</s>"])
        padded_vocab = Vocabulary(words)
        print(words)
        self.lm.fit(train_data, padded_vocab)


    def scoreQuestions(self, listofquestions):

        # need to add terminal steps from pos tages to single words
        tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in listofquestions]
        print(tokenized_text)
        test_data = [list(nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>")) for
                     t in tokenized_text]
        results = []
        print(test_data)
        for test in test_data:
            print(self.lm.entropy(test))
            results.append(self.lm.entropy(test))
            print("MLE Estimates:", [((ngram[-1], ngram[:-1]), self.lm.unmasked_score(ngram[-1], ngram[:-1])) for ngram in test])
        return results

test = QuestionScorer("Birds of Prey (and the Fantabulous Emancipation of One Harley Quinn) is a 2020 American superhero film based on the DC Comics team Birds of Prey. It is the eighth film in the DC Extended Universe and a follow-up to Suicide Squad (2016). It was directed by Cathy Yan and written by Christina Hodson, and it stars Margot Robbie, Mary Elizabeth Winstead, Jurnee Smollett-Bell, Rosie Perez, Chris Messina, Ella Jay Basco, Ali Wong, and Ewan McGregor. The film follows Harley Quinn as she joins forces with Helena Bertinelli, Dinah Lance, and Renee Montoya to save Cassandra Cain from Gotham City crime lord Black Mask.")
print(test.scoreQuestions(["Birds of Prey.", "Cathy Yan."]))