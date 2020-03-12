import StringProcessor
import QuestionScorer
if __name__ == '__main__':
    stringprocessor = StringProcessor()
    questionscorer = QuestionScorer()
    numQs = 5 # return top 5 questions
    with open('data.txt', 'r') as file:
        data = file.read().replace('\n', ' ')
    tokens = stringprocessor.tokenize(data)
    # POS key: https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b
    POS = stringprocessor.pos_tag(tokens)
    questions = []
    ### Template Matching
    for i in range(0, len(POS)):
        (word, POS) = POS[i]
        if word == 'is' and i > 0 and i < len(POS)-1: # Template for X is Y
            # TODO instead of just using adjacent words, we can try using dependent words from a dependency tree
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
        ## TODO add more templates here


    ### Grammar Parser

    ### Question Scorer
    scores = questionscorer.scoreQuestions(questions)
    sortedscores = sorted(scores, key=lambda x: x[1]) # sort by score
    for i in range(0, numQs):
        (Q,_) = sortedscores[i]
        print(Q+ '\n')