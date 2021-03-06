#!/usr/bin/env python3
import StringProcessor
import sys
import stanfordnlp
import re
import io
text_trap = io.StringIO()
sys.stdout = text_trap
sys.stderr = text_trap

Questions = ["Between 2007 and 2012 , who played for Premier League team Fulham and is the club 's highest Premier League goalscorer of all time?", "Is Dempsey of Irish descent on his father 's side?"]
wh_set = {"who", "what", "which", "where"}
non_d_set = {"who", "what", "which", "where", "when", "that"}
yes_no_set = {"is", "was", "are", "were"}
yes_no_check = relation_list = {"nsubj", "obj", "nummod", "iobj", "nsubj_pass"}

nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
def dependency_parse(sentence):
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
        w, i, r, c = l[0], l[1], l[2], l[3]
        if len(c) > 0:
            w = ""
            for i in sorted(c):
                w += parsed_list[i][0]
                w += " "
            w = w[:-1]
        ret.append((w, i, r))
    return ret

def match_yes_no_parts(q_dep, ans_dep):
    ans = "Yes"
    for (w, _, d) in q_dep:
        if d in yes_no_check:
            found = False
            for (a, _, b) in ans_dep:
                if b == d:
                    if a.lower() == w.lower():
                        found = True
            if not found:
                ans = "No"
    return ans

if __name__ == '__main__':
    args = sys.argv
    with open(args[2], 'r') as file:
        Questions = file.readlines()
    with open(args[1], 'r') as file:
        data = file.read().replace('\n', ' ')
    
    sentences2 = StringProcessor.sent_tokenize(data)
    data = StringProcessor.coreference(data)
    sentences = StringProcessor.sentence_tokenize(data)
    tok_sent = [StringProcessor.tokenize(x) for x in sentences]
    ners_sent = [StringProcessor.old_NER(x) for x in tok_sent] #not sure how this works
    ners_set = []
    for i in range(0,len(ners_sent)):
        ners_set.append([])
        for x in ners_sent[i].subtrees():
            for y in x.subtrees():
                if(y.label() != 'S'):
                    ners_set[i].append(y)
    answerlist = []

    for q in Questions:
        yes_no1 = True
        ques_sent = StringProcessor.tokenize(q)
        for (i, x) in enumerate(ques_sent):
            if x.lower() in wh_set:
                wh_pos = i
                yes_no1 = False
        yes_no2 = ques_sent[0].lower() in yes_no_set

        ques_ner = StringProcessor.old_NER(ques_sent)
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
        index = goodq[cos_sims.index(max(cos_sims))]
        ans_sent = sentences[index]
        ans_sent2 = sentences2[index]

        q_dep = dependency_parse(q)
        ans_dep = dependency_parse(ans_sent)
        # print(q_dep)
        ans_dep2 = dependency_parse(ans_sent2)
        dep = None
        root = None
        if not yes_no1 and not yes_no2:
            ans = []
            for (w, i, d) in q_dep:
                if w.lower() in wh_set:
                    if d == "root":
                        dep = "nsubj"
                    else:
                        dep = d
                elif d == "nsubj" or d == "root":
                    root = w.lower()
            # answerlist.append(ans_dep)
            for (w, i, d) in ans_dep:
                if d == dep and w not in non_d_set: 
                    ans.append(w.lower())
            answerlist.append(" ".join(ans) + ".")
        else:
            ans = match_yes_no_parts(q_dep, ans_dep2)
            answerlist.append(ans)

    sys.stdout = sys.__stdout__
    #for i in range(0,)
    for answer in answerlist:
        print(answer)








