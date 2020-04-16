import StringProcessor as sp

def inlist(a,l):
    for i in l:
        if a==i:
            return True
    return False


def GenerateYesNo(sentence):
    startpos = 0
    isloc = -1
    endloc = len(sentence)-1
    for i in range(0,len(sentence)):
        if sentence[i]==',':
            if(isloc>=0):
                if inlist(sentence[i+1],['and','or','not','so','although']):
                    endloc = i
                elif (not inlist(sentence[i+1],['while','if'])) and inlist(sentence[0],['If','Since','Although','While','When','What','Whatever']):
                    isloc = -1
                    startpos = i+1

            else:
                startpos = i+1
        elif inlist(sentence[i],['is','was','are','were','can','could','should','must']):
            if not inlist(sentence[i-1],['this','This','that','That','he','she', 'it','He','She','It','They','they']):
                if(isloc < 0):
                    isloc = i
    if(isloc < 0):
        return None
    out = ""
    for i in range(0,startpos):
        out += (" "+sentence[i])
    out += sentence[isloc]
    for i in range(startpos,endloc):
        if not(i==isloc):
            out += (" "+sentence[i])
    return out + "?"

#print(GenerateYesNo(['Rose','is','red','.']))
#print(GenerateYesNo(['In','1998',',','Tom','was','11','years','old','.']))
#print(GenerateYesNo(sp.tokenize("If I can do a flip, I can drop the course.")))
#print(GenerateYesNo(sp.tokenize("Michael is very smart, and Andy is very smart too.")))
#print(GenerateYesNo(sp.tokenize("Dempsey was born in Nacogdoches, Texas, and, for much of his childhood, his family lived in a trailer park,"
#                                " where he and his siblings grew up playing soccer with Hispanic immigrants.")))












