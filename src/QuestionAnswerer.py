import StringProcessor

Questions = ["What is Carnegie Mellon University?", "What did Carnegie Institute of Technology merge with in 1967?"]

if __name__ == '__main__':
    with open('../data/data.txt', 'r') as file:
        data = file.read().replace('\n', ' ')
    sentences = StringProcessor.sentence_tokenize(data)
    tok_sent = [StringProcessor.tokenize(x) for x in sentences]
    ners_sent = [StringProcessor.NER(x) for x in tok_sent]
    ners_set = []
    for i in range(0,len(ners_sent)):
        ners_set.append([])
        for x in ners_sent[i].subtrees():
            for y in x.subtrees():
                if(y.label() != 'S'):
                    ners_set[i].append(y)
    answerlist = []

    for q in Questions:
        ques_sent = StringProcessor.tokenize(q)
        ques_ner = StringProcessor.NER(ques_sent)
        qners = []
        for x in ques_ner.subtrees():
            for y in x.subtrees():
                if(y.label() != 'S'):
                    qners.append(y)
        #filter by NER contained
        goodq = []
        for i in range(0,len(ners_set)):
            sent = ners_set[i]
            good = True
            for qner in qners:
                appeared = False
                for ner in sent:
                    if(ner == qner):
                        appeared = True
                        break
                if not appeared:
                    good=False
                    break
            if good:
                goodq.append(i)
        if len(goodq) == 0:
            goodq = [i for i in range(0,len(ners_sent))]
        #further filter by match similarity
        cos_sims = [StringProcessor.match_similarity(ques_sent, tok_sent[i]) for i in goodq]
        index = cos_sims.index(max(cos_sims))
        answerlist.append(sentences[index])
    print(answerlist)








