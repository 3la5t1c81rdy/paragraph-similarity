"""
Field where "main" results are produced
Also generates raw paraphrase pairs and saves them in a paginated file format
    Gen_Paraphrases and Gen_N_Paraphrases
    

Uses the paraphrase dataset to in turn run predictors and generate results
"""
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
CT_ALECS = 101895
####
def Gen_Paraphrases(p = 0, page_size = 1000):
    #ignore p ("initial page number") and page_size ("page size"); used for debug purposes.
    #collects matching pairs from the Datasets, and stores them away in a more convenient
    #paginated chunks of (at most) 1000 pairs ... instructs <D>_collect_matches to LOOK at 1000 file 
    #    pairs at once
    #
    #Saves to PICKLES_DIR/<W/A>-pairs-<#page>.cpkl
    
    w = ["!"]
    a = ["!"]
    while True:
        w = Webis_collect_matches(WEBIS_DIR, page=p, page_size=page_size)
        a = ALECS_collect_matches(ALECS_DIR_OG, ALECS_DIR_MG, page=p, page_size=page_size)
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
        
def _Webis_Negative_Paraphrases(n = 50000) -> list[(str, str, os.path, os.path)]:
    #generates 50000 negative paraphrases (sourced from POSITIVES in Webis)
    #this function belongs in Main because this function requires [Gen_Paraphrases] to have run
    #    "non-paraphrases" as identified in the original Webis dataset includes both
    #    insufficient paraphrasing AND non-paraphrasing (+I have inspected it, so they should not
    #        belong in the testing process anyways)
    wpath = ""
    if not os.path.isfile(wpath := os.path.join(PICKLES_DIR, "W-pairs-0.cpkl")):
        raise FileNotFoundError(f"{wpath} must be a file; to generate negative paraphrases from Webis-CPC dataset, run 'Gen_Paraphrases' first.")
    
    #discover all W-pairs
    wpath = ["W-pairs-0.cpkl"]
    i = 1
    while os.path.isfile(os.path.join(PICKLES_DIR, f"W-pairs-{i}.cpkl")):
        wpath.append(f"W-pairs-{i}.cpkl")
        i += 1
    
    pos = []
    for w in wpath:
        pos += Recover_Object(os.path.join(PICKLES_DIR, w))
    
    rl = []
    for _ in range(n):
        x, i = random.randint(0, len(pos) - 1), random.randint(0, 1)
        j = random.randint(0, 1)
        while (y := random.randint(0, len(pos) - 1)) == x:
            #y must be different from x
            continue
        
        rl.append((pos[x][i], pos[y][j], pos[x][2+i], pos[y][2+j]))
    
    return rl

def Gen_N_Paraphrases(p = 0, page_size = 1000, n_each = CT_ALECS):
    #Analogous to Gen_Paraphrases above, except the stored pairs are NEGATIVE paraphrases
    #by default generates the negative samples from each of the two datasets, of counts equalling
    #    the number of positive pairs in the largest dataset (ALECS)
    #        It was discovered to be 101895.
    #
    #p is still largely for debug purposes
    #... i.e. two paragraphs are strictly NOT the paraphrases provided from each of Webis and 
    #    ALECS datasets
    #    (some of the pairs may coincidentally have similar semantic meanings, but the assumption 
    #    is that the ratio of such cases is sufficiently small)
    #paginated chunks of 1000 text pairs
    #
    #Saves to PICKLES_DIR/<W/A>-negative-pairs-<#page>.cpkl
    
    if p < 0 or page_size <= 0 or n_each <= (p + 1) * page_size:
        return
    max_n = n_each
    n_each -= p * page_size
    A_Negatives = ALECS_pick_non_match(ALECS_DIR_OG, ALECS_DIR_MG, n_each)
    W_Negatives = _Webis_Negative_Paraphrases(n_each)
    
    while p < ((max_n - 1)//page_size) + 1:
        a = A_Negatives[p * page_size : (p + 1) * page_size]
        w = W_Negatives[p * page_size : (p + 1) * page_size]
        Save_Object(a, os.path.join(PICKLES_DIR, f"A-negative-pairs-{p}.cpkl"))
        Save_Object(w, os.path.join(PICKLES_DIR, f"W-negative-pairs-{p}.cpkl"))
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
        if "negative" in pf:
            continue
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
        if "negative" in pf:
            continue
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

def Predict_A_Negative_SEF1(b = 1000, n = 100):
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
        #Save_Object([(a, b) for (_, _, a, b) in batch], os.path.join(RES_NEG_DIR, f"IDX_A-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in batch], os.path.join(RES_NEG_DIR, f"IDX_SE_F1_A-{i}.cpkl"))

def Predict_A_Negative_BS(b = 1000, n = 100):
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

def Predict_A_Negative_BLEU(b = 1000, n = 100):
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
   
def Predict_W_Negative():
    paraphrase_files = os.listdir(PICKLES_DIR)
    
    BLEU = B()
    print("BLEU", time.time())
    SE_F1 = SF()
    print("SF", time.time())
    
    i = 0
    for pf in paraphrase_files:
        if pf.find("W-negative-pairs-") != 0:
            #pick out W-negative
            continue
        print(f"Predicting {pf}")
        curr = os.path.join(PICKLES_DIR, pf)
        curr = Recover_Object(curr)
        
        refs = [x for (x, _, _, _) in curr]
        cands = [x for (_, x, _, _) in curr]
        
        r_bleu = BLEU.score(cands, refs)
        print("BLEU", time.time())
        
        """ #bertscore is particularly resource intensive. Separating it out...
        r_bertscore = BERTScore.score(cands, refs)
        print("BERTScore", time.time())
        """
        
        r_sef1 = SE_F1.score(cands, refs, info=False)
        print("SE-F1", time.time())
        
        Save_Object(r_bleu, os.path.join(RES_NEG_DIR, f"BLEU_W-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in curr], os.path.join(RES_NEG_DIR, f"IDX_BLEU_W-{i}.cpkl"))
        """
        Save_Object(r_bertscore, os.path.join(RES_NEG_DIR, f"BERTSCORE_W-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in curr], os.path.join(RES_NEG_DIR, f"IDX_BS_W-{i}.cpkl"))
        """
        Save_Object(r_sef1, os.path.join(RES_NEG_DIR, f"SE_F1_W-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in curr], os.path.join(RES_NEG_DIR, f"IDX_SE_F1_W-{i}.cpkl"))
        
        i += 1

   
def Predict_W_Negative_BS():
    #BERTScore only.
    paraphrase_files = os.listdir(PICKLES_DIR)
    
    BERTScore = BS()
    print("BS", time.time())
    
    i = 0
    for pf in paraphrase_files:
        if pf.find("W-negative-pairs-") != 0:
            #pick out W-negative
            continue
        print(f"Predicting {pf}")
        curr = os.path.join(PICKLES_DIR, pf)
        curr = Recover_Object(curr)
        
        refs = [x for (x, _, _, _) in curr]
        cands = [x for (_, x, _, _) in curr]
        
        #bertscore is particularly resource intensive. Separating it out...
        r_bertscore = BERTScore.score(cands, refs)
        print("BERTScore", time.time())
        
        Save_Object(r_bertscore, os.path.join(RES_NEG_DIR, f"BERTSCORE_W-{i}.cpkl"))
        Save_Object([(a, b) for (_, _, a, b) in curr], os.path.join(RES_NEG_DIR, f"IDX_BS_W-{i}.cpkl"))
        
        i += 1

if __name__ == "__main__":
    """# run to populate a directory of paraphrases in PICKLES_DIR
    Gen_Paraphrases(p=0)
    """
    
    """#run to populate a directory of negative paraphrases in PICKLES_DIR
    Gen_N_Paraphrases(p=0)
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
    Predict_A_Negative_SEF1() #100000 (negative) pairs
    #mistake: SEF1's indices are in IDX_A_<>, not IDX_SEF1_A_<>
    """
    """
    #^^^ remedy: pick out all IDX_A_ files and change them into IDX_SEF1_A_ files
    fns = os.listdir(RES_NEG_DIR)
    for f in fns:
        if f.find("IDX_A") != 0:
            continue
        tail = f[len("IDX_A"):]
        rename = "IDX_SE_F1_A" + tail
        print(f"rename {os.path.join(RES_NEG_DIR, f)} into {os.path.join(RES_NEG_DIR, rename)}")
        os.rename(os.path.join(RES_NEG_DIR, f), os.path.join(RES_NEG_DIR, rename))
    """
    """
    Predict_A_Negative_BS() #100000 (negative) pairs for BS
    Predict_A_Negative_BLEU() #100000 (negative) pairs for BLEU
    """
    #
    """
    Predict_W_Negative() #101895 (negative) pairs for BLEU, SE-F1
    Predict_W_Negative_BS() #Same for BERTScore
    """