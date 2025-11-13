from Utils.Utils import Save_Object, Recover_Object, \
    Webis_collect_matches, ALECS_collect_matches, ALECS_pick_non_match
import os
from Predictors.BERTScore import BERTScore as BS
from Predictors.BLEU import BLEU as B
from Predictors.SE_F1 import SE_F1 as SF
import time
import random
import gc
import torch

####
ALECS_DIR_OG = "Datasets/ALECS/og"
ALECS_DIR_MG = "Datasets/ALECS/mg"
WEBIS_DIR = "Datasets/Webis-CPC-11"
PICKLES_DIR = "Datasets/Paraphrases"
RES_POS_DIR = "./Predictions/Positives/"
RES_NEG_DIR = "./Predictions/Negatives/"
####
def Gen_Paraphrases(p = 0):
    w = ["!"]
    a = ["!"]
    while True:
        w = Webis_collect_matches(WEBIS_DIR, page=p, page_size = 1000)
        a = ALECS_collect_matches(ALECS_DIR_OG, ALECS_DIR_MG, page=p, page_size = 1000)
        if len(w) == 0 and len(a) == 0:
            break
        if len(w) > 0:
            Save_Object(w, os.path.join(PICKLES_DIR, f"W-pairs-{p}.cpkl"))
        if len(a) > 0:
            Save_Object(a, os.path.join(PICKLES_DIR, f"A-pairs-{p}.cpkl"))
        w.clear()
        a.clear()
        print(f"page {p}")
        p += 1

def Predict_Positive():
    paraphrase_files = os.listdir(PICKLES_DIR)
    BLEU = B()
    print("BLEU", time.time())
    """
    BERTScore = BS()
    print("BS", time.time())
    """
    SE_F1 = SF()
    print("SF", time.time())
    for pf in paraphrase_files:
        print(pf)
        curr = os.path.join(PICKLES_DIR, pf)
        curr = Recover_Object(curr)
        
        refs = [x for (x, _, _, _) in curr]
        cands = [x for (_, x, _, _) in curr]
        r_bleu = BLEU.score(cands, refs)
        print("BLEU", time.time())
        """
        i = 0
        r_bertscore = []
        while i < len(refs):
            r_bertscore.append(BERTScore.score([cands[i]], [refs[i]]))
            i += 1
            if i%10 == 0: print(i, time.time())
        print("BERTScore", time.time())
        """
        
        r_sef1 = SE_F1.score(cands, refs, info=False)
        print("SE_F1", time.time())
        Save_Object(r_bleu, os.path.join(RES_POS_DIR, f"BLEU_{pf}"))
        Save_Object(r_sef1, os.path.join(RES_POS_DIR, f"SE_F1_{pf}"))

def Predict_Positive_BS():
    
    paraphrase_files = os.listdir(PICKLES_DIR)
    BERTScore = BS()
    print("BS", time.time())
    for pf in paraphrase_files:
        print(pf)
        curr = os.path.join(PICKLES_DIR, pf)
        curr = Recover_Object(curr)
        
        refs = [x for (x, _, _, _) in curr]
        cands = [x for (_, x, _, _) in curr]
        
        r_bertscore = BERTScore.score(cands, refs)
        print("BERTScore", time.time())
        Save_Object(r_bertscore, os.path.join(RES_POS_DIR, f"BERTSCORE_{pf}"))
        
        del refs, cands, r_bertscore
        gc.collect()
        torch.cuda.memory.empty_cache()

def Predict_Negative_SEF1(b = 1000, n = 100):
    #b * n pairs to be graded
    """
    BLEU = B()
    print("BLEU", time.time())
    BERTScore = BS()
    print("BS", time.time())
    """
    SE_F1 = SF()
    print("SF", time.time())
    
    for i in range(n):
        print(f"Batch {i}")
        batch = ALECS_pick_non_match(ALECS_DIR_OG, ALECS_DIR_MG, b)
        print("loaded")
        refs = [x for (x, _, _, _) in batch]
        cands = [x for (_, x, _, _) in batch]
        
        r_sef1 = SE_F1.score(cands, refs, info=False)
        print("SE_F1", time.time())
        Save_Object(r_sef1, os.path.join(RES_NEG_DIR, f"SE_F1_A-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in batch], os.path.join(RES_NEG_DIR, f"IDX_A-{i}.cpkl"))

def Predict_Negative_BS(b = 1000, n = 100):
    #b * n pairs to be graded
    BERTScore = BS()
    print("BS", time.time())
    
    for i in range(n):
        print(f"Batch {i}")
        batch = ALECS_pick_non_match(ALECS_DIR_OG, ALECS_DIR_MG, b)
        print("loaded")
        refs = [x for (x, _, _, _) in batch]
        cands = [x for (_, x, _, _) in batch]
        r_bertscore = BERTScore.score(cands, refs)
        print("BS", time.time())
        
        Save_Object(r_bertscore, os.path.join(RES_NEG_DIR, f"BERTSCORE_A-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in batch], os.path.join(RES_NEG_DIR, f"IDX_BS_A-{i}.cpkl"))


def Predict_Negative_BLEU(b = 1000, n = 100):
    #b * n pairs to be graded
    BLEU = B()
    print("BLEU", time.time())
    
    for i in range(n):
        print(f"Batch {i}")
        batch = ALECS_pick_non_match(ALECS_DIR_OG, ALECS_DIR_MG, b)
        print("loaded")
        refs = [x for (x, _, _, _) in batch]
        cands = [x for (_, x, _, _) in batch]
        r_bleu = BLEU.score(cands, refs)
        print("BLEU", time.time())
        
        Save_Object(r_bleu, os.path.join(RES_NEG_DIR, f"BLEU_A-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in batch], os.path.join(RES_NEG_DIR, f"IDX_BLEU_A-{i}.cpkl"))
        

if __name__ == "__main__":
    """# run to populate a directory of paraphrases in PICKLES_DIR
    Gen_Paraphrases(p=0)
    """
    """
    Predict_Positive() #run (positive) predictions (SEF1, BLEU)
    """
    """
    Predict_Positive_BS() #(positive) predictions for BERTScore
    """
    """
    # below is a test code to reassure the number of responses on positive predictions
    L = os.listdir(RES_POS_DIR)
    c1, c2 = 0, 0
    for n in L:
        ni = os.path.join(RES_POS_DIR, n)
        if n.find("BLEU_A") == 0:
            p = Recover_Object(ni)
            c1 += len(p)
        if n.find("SE_F1_A") == 0:
            p = Recover_Object(ni)
            c2 += len(p)
    
    print(c1, c2) #101895 pairs as expected from ALECS
    """
    """
    Predict_Negative_SEF1() #100000 (negative) pairs
    #mistake: SEF1's indices are in IDX_A_<>, not IDX_SEF1_A_<>
    """
    Predict_Negative_BS() #100000 (negative) pairs for BS
    Predict_Negative_BLEU() #100000 (negative) pairs for BLEU