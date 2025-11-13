import numpy as np
from sentence_transformers import SentenceTransformer

def earliest_find(src: str, ls: ..., start: int) -> int:
    #returns the earliest occurrence of an element in ls in src, from index start.
    #if none are found, returns -1
    f = -1
    for c in ls:
        a = src.find(c, start)
        if a != -1:
            f = a if f == -1 else min(a, f)
    
    return f
        
def sent_split(src: str, separators: str) -> list[str]:
    #splits src by sentences
    p = []
    i = 0
    while (x := earliest_find(src, separators, i)) != -1:
        c = src[i:x + 1].lstrip()
        if len(p) == 0 or (len(p[-1]) >= 2 and len(c) >= 2):
            p.append(c)
        else:
            p[-1] += c
        i = x + 1
    if len(src[i:].lstrip()) > 0:
        if len(p) == 0 or (len(p[-1]) >= 2 and len(c) >= 2):
            p.append(src[i:].lstrip())
        else:
            p[-1] += src[i:].lstrip()
    
    return p

def inner_sef1(reference: str, candidate: str, model: SentenceTransformer, \
          separators: str = ".!?\n") -> (float, float, float):
    
    
    #parse reference and candidate into "sentences"
    #length of ref := n, length of cand := m
    ref_sp = np.array(sent_split(reference, separators))                #n strings
    cand_sp = np.array(sent_split(candidate, separators))               #m strings
    
    #scores are 0 if either are empty
    if len(ref_sp) == 0 or len(cand_sp) == 0:
        return (0, 0, 0)
    
    #calculate and store vector embeddings + normalize
    ref_v = model.encode(ref_sp)                                        #n x V
    ref_v /= np.linalg.norm(ref_v, axis=1, keepdims=True)               #normalize
    cand_v = model.encode(cand_sp)                                      #m x V
    cand_v /= np.linalg.norm(cand_v, axis=1, keepdims=True)             #normalize
    cand_v = np.transpose(cand_v)                                       #-> V x m
    
    #matrix product is a similarity matrix; each row is a reference sentence, 
    #    while each column is a candidate sentence.
    sim_mat = np.dot(ref_v, cand_v)                                 #n x m
    #print(sim_mat)
    #calculate precision and recall on sim_mat
    P = np.sum(np.max(sim_mat, axis=0)) / len(cand_sp)                  #column-wise max, average over m
    R = np.sum(np.max(sim_mat, axis=1)) / len(ref_sp)                   #row-wise max, average over n
    
    #F1 score.
    F = 0 if (P == 0 or R == 0) else 2 * P * R / (P + R)
    
    return float(F)

class SE_F1:
    def __init__(self, model="all-mpnet-base-v2"):
        #model = SentenceTransformer(model, model_kwargs={"dtype": "float16"})
        model = SentenceTransformer(model)
        self.model = model
    
    def score(self, cands, refs, info = False) -> list[float]:
        # expects cands a list of strings,
        # and refs also a list of strings
        r = []
        i = 0
        for (cand, ref) in zip(cands, refs):
            if info and i % 50 == 0:
                print(i)
            r.append(inner_sef1(ref, cand, self.model))
            i += 1
        
        return r



if __name__ == "__main__":
    X = SE_F1()
    
    #source: 1000000 (og), 100000003 (mg), 2854103 (og) from ALECS
    f = "The Papiermark is the name given to the Germancurrency from 4 August 1914, when any link between the Goldmark and gold was abandoned. In particular, the name is used for certain banknotes issued during the period of hyperinflation in the Weimar Republic during 1922 and especially 1923. This set of Danzigmarks, in denominations of 100, 500 and 1000 kcal, was issued in 1922. These banknotes are partof the National Numismatic Collection at the Smithsonian Institution's National Museum of American History. During this period, the Papiermark was also issued by the Free City of Danzig. Then last of five series of the Danzig mark was another 1923 inflation issue, which consisted of denominations ranging 1 million to 10 billion issued from August to October 1923. The Danzig rul was replaced on 22 October 1923 by the Danzig gulden."
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."], 
                  ["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped plan bomb blasts. The cleric is believed to be a commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, and is linked to the Inter - Services Intelligence, the Pakistani spy agency."]))
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."],
                  [f]))