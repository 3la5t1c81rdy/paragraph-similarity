import numpy as np
import torch
import time
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
          separators: str = ".!?\n") -> float:
    
    
    #parse reference and candidate into "sentences"
    #length of ref := n, length of cand := m
    ref_sp = np.array(sent_split(reference, separators))                #n strings
    cand_sp = np.array(sent_split(candidate, separators))               #m strings
    
    #scores are 0 if either are empty
    if len(ref_sp) == 0 or len(cand_sp) == 0:
        return 0
    
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
    #calculate precision and recall on sim_mat (treating each <0 "as" 0)
    P = np.sum(np.maximum(0,np.max(sim_mat, axis=0))) / len(cand_sp)    #column-wise max, average over m
    R = np.sum(np.maximum(0,np.max(sim_mat, axis=1))) / len(ref_sp)     #row-wise max, average over n
    
    #F1 score.
    F = 0 if (P == 0 or R == 0) else 2 * P * R / (P + R)
    
    return float(F)


def torch_inner_sef1(reference: str, candidate: str, model: SentenceTransformer, \
          separators: str = ".!?\n") -> float:
    #parse reference and candidate into "sentences"
    #length of ref := n, length of cand := m
    ref_sp = model.tokenize(sent_split(reference, separators))                #n strings
    cand_sp = model.tokenize(sent_split(candidate, separators))               #m strings
    
    #scores are 0 if either are empty
    if len(ref_sp["input_ids"]) == 0 or len(cand_sp["input_ids"]) == 0:
        return torch.tensor(0, requires_grad=True)
    
    #calculate and store vector embeddings + normalize
    ref_v = model(ref_sp)["sentence_embedding"]                   #n x V
    ref_v = ref_v / torch.norm(ref_v, dim=1, keepdim=True)               #normalize
    cand_v = model(cand_sp)["sentence_embedding"]                 #m x V
    cand_v = cand_v / torch.norm(cand_v, dim=1, keepdim=True)             #normalize
    cand_v = torch.transpose(cand_v, 0, 1)                        #-> V x m
    
    #matrix product is a similarity matrix; each row is a reference sentence, 
    #    while each column is a candidate sentence.
    sim_mat = torch.matmul(ref_v, cand_v)                                 #n x m
    #print(sim_mat)
    #calculate precision and recall on sim_mat (treating each <0 "as" 0)
    P = torch.sum(torch.clamp(torch.max(sim_mat, dim=0)[0], min=0)) / len(cand_sp["input_ids"])    #column-wise max, average over m
    R = torch.sum(torch.clamp(torch.max(sim_mat, dim=1)[0], min=0)) / len(ref_sp["input_ids"])   #row-wise max, average over n
    
    #F1 score.
    F = 2 * P * R / (P + R + 1e-10)
    return F

class SE_F1:
    def __init__(self, model="all-mpnet-base-v2"):
        #model = SentenceTransformer(model, model_kwargs={"dtype": "float16"})
        if type(model) is str:
            model = SentenceTransformer(model)
        self.model = model
    
    def score(self, cands, refs, info = False) -> list[float]:
        # expects cands a list of strings,
        # and refs also a list of strings
        r = []
        i = 0
        for (cand, ref) in zip(cands, refs):
            if info and i % 50 == 0:
                print(i, int(time.time()*10)/10, end=" || ")
                if i % 500 == 0:
                    print()
            r.append(inner_sef1(ref, cand, self.model))
            i += 1
        return r
    
    def _torch_score(self, cands, refs, info = False) -> torch.tensor:
        #used for autograd purposes
        r = torch.tensor([])
        i = 0
        for (cand, ref) in zip(cands, refs):
            if info and i % 50 == 0:
                print(i, int(time.time()*10)/10, end=" || ")
                if i % 500 == 0:
                    print()
            r = torch.cat((r, torch_inner_sef1(ref, cand, self.model).expand(1)))
            i += 1
        return r

class CalibratedSE_F1(SE_F1):
    #SE_F1 with model and all, given a calibration map.
    #***Manually fit the calibration mapping beforehand, using Utils.Fit_Calibration()!!***
    def __init__(self, model: "str | SentenceTransformer", calibration_mapping, disable_score = False):
        #set disable_score to True if you only want to map/maps the pre-existing predictions and not .score() directly.
        if not disable_score:
            super().__init__(model)
            self._disable_score = False
        else:
            self._disable_score = True
        if not "map" in dir(calibration_mapping):
            raise ValueError(f"calibration_mapping {calibration_mapping} does not support .map(f_x). Ensure it is a RangeMap (or alike) that can map a prediction into a calibrated prediction.")
        self.rm = calibration_mapping
    
    def score(self, cands, refs, info=False)->list[float]:
        if self._disable_score:
            raise ValueError(f"{self} has not been configured to be able to directly score!")
        return [self.rm.map(x) for x in SE_F1.score(self, cands, refs, info=info)]
    
    def map(self, f_x):
        #a convenience (... if the prediction is already given, simply pass to the self.rm)
        return self.rm.map(f_x)
    
    def maps(self, ls_f_x: list[float]):
        #also a convenience method. This time maps each item in ls_f_x.
        return [self.rm.map(f_x) for f_x in ls_f_x]

if __name__ == "__main__":
    import time
    X = SE_F1()
    #source: 1000000 (og), 100000003 (mg), 2854103 (og) from ALECS
    print(time.time())
    f = "The Papiermark is the name given to the Germancurrency from 4 August 1914, when any link between the Goldmark and gold was abandoned. In particular, the name is used for certain banknotes issued during the period of hyperinflation in the Weimar Republic during 1922 and especially 1923. This set of Danzigmarks, in denominations of 100, 500 and 1000 kcal, was issued in 1922. These banknotes are partof the National Numismatic Collection at the Smithsonian Institution's National Museum of American History. During this period, the Papiermark was also issued by the Free City of Danzig. Then last of five series of the Danzig mark was another 1923 inflation issue, which consisted of denominations ranging 1 million to 10 billion issued from August to October 1923. The Danzig rul was replaced on 22 October 1923 by the Danzig gulden."
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."], 
                  ["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped plan bomb blasts. The cleric is believed to be a commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, and is linked to the Inter - Services Intelligence, the Pakistani spy agency."]))
    
    print(time.time())
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."],
                  [f]))
    """
    print(X._torch_score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."], 
                  ["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped plan bomb blasts. The cleric is believed to be a commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, and is linked to the Inter - Services Intelligence, the Pakistani spy agency."]))
    
    print(time.time())
    
    print(X._torch_score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."],
                  [f]))
    """