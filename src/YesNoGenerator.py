import StringProcessor as sp

def inlist(a,l):
    for i in l:
        if a.lower()==i.lower():
            return True
    return False

def GenerateYesNo(sentence):
    if inlist(',',sentence) or inlist('and',sentence) or inlist('when',sentence):
        return None
    startpos = 0
    isloc = -1
    endloc = len(sentence)-1
    prevfirst = 0
    for i in range(0,len(sentence)):
        if sentence[i]==',':
            if(isloc>=0) and (isloc < i-1):
                if inlist(sentence[i+1],['and','or','not','so','although']):
                    endloc = i
                elif (not inlist(sentence[i+1],['while','if'])) and inlist(sentence[prevfirst],['If','Since','Although','While','When','What','Whatever']):
                    isloc = -1
                    startpos = i+1

            else:
                if(isloc==i-1):
                    isloc = -1
                    break
                startpos = i+1
            prevfirst = i + 1
        elif inlist(sentence[i],['is','was','are','were','can','could','should','must']):
            if not inlist(sentence[i-1],['this','that','he','she', 'it','they','and','so','or','either','which','who','what','where']):
                if(isloc < 0):
                    isloc = i
    if(isloc < 0) or (isloc - startpos > 9):
        return None
    out = ""
    sentence[0] = sentence[0][0].lower()+sentence[0][1:]
    for i in range(0,startpos):
        out += (" "+sentence[i])
    out += sentence[isloc]
    for i in range(startpos,endloc):
        if not(i==isloc):
            out += (" "+sentence[i])
    return out + "?"
"""
print(GenerateYesNo(['Rose','is','red','.']))
print(GenerateYesNo(['In','1998',',','Tom','was','11','years','old','.']))
print(GenerateYesNo(sp.tokenize("If I can do a flip, I can drop the course.")))
print(GenerateYesNo(sp.tokenize("Michael is very smart, and Andy is very smart too.")))
print(GenerateYesNo(sp.tokenize("Dempsey was born in Nacogdoches, Texas, and, for much of his childhood, his family lived in a trailer park,"
                                " where he and his siblings grew up playing soccer with Hispanic immigrants.")))
print(GenerateYesNo(sp.tokenize("Given enough information, if I can do a flip, I can drop the course.")))

"""









