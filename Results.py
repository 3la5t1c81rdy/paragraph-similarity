# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
from Utils.Utils import Recover_Object

##
RES_POS_DIR = "./Predictions/Positives/"
RES_NEG_DIR = "./Predictions/Negatives/"
PREF_BLEU = "BLEU_"
PREF_BERTSCORE = "BERTSCORE_"
PREF_SE_F1 = "SE_F1_"
PREF_W = "W-"
PREF_A = "A-"
##

def Aggr_Positives():
    #(<Webis>, <ALECS>) where each are list of lists of strings, of results
    #... in order of BLEU, BERTSCORE, SE_F1
    Aggr = ([[],[],[]],[[],[],[]])
    L = os.listdir(RES_POS_DIR)
    for fn in L:
        if fn.find(PREF_BLEU) == 0:
            t = fn[len(PREF_BLEU):]
            j = 0
        elif fn.find(PREF_BERTSCORE) == 0:
            t = fn[len(PREF_BERTSCORE):]
            j = 1
        elif fn.find(PREF_SE_F1) == 0:
            t = fn[len(PREF_SE_F1):]
            j = 2
        else:
            continue
        if t.find(PREF_W) == 0:
            i = 0
        elif t.find(PREF_A) == 0:
            i = 1
        else:
            continue
        
        curr = Recover_Object(os.path.join(RES_POS_DIR, fn))
        Aggr[i][j] += curr
        
    
    return Aggr

def Aggr_Negatives():
    #Negatives only contain ALECS-sourced pairs.
    #Returns a list of results in order of BLEU, BERTSCORE, SE_F1
    Aggr = [[],[],[]]
    L = os.listdir(RES_NEG_DIR)
    for fn in L:
        if fn.find(PREF_BLEU) == 0:
            t = fn[len(PREF_BLEU):]
            j = 0
        elif fn.find(PREF_BERTSCORE) == 0:
            t = fn[len(PREF_BERTSCORE):]
            j = 1
        elif fn.find(PREF_SE_F1) == 0:
            t = fn[len(PREF_SE_F1):]
            j = 2
        else:
            continue
        if t.find(PREF_A) == 0:
            i = 0
        else:
            print(fn, "??")
            continue
        
        curr = Recover_Object(os.path.join(RES_NEG_DIR, fn))
        Aggr[j] += curr
        
    
    return Aggr

if __name__ == "__main__":
    
    P = Aggr_Positives()
    N = Aggr_Negatives()
    print(len(P[0][0]))
    print(len(P[1][0]))
    #print(N[0])
    plt.figure(0)
    plt.title("Milestone results - Positive pairs, BLEU, Webis (n = 4067)")
    plt.xlabel("BLEU score (\u0394 = 0.01)")
    plt.ylabel("Frequencies")
    plt.hist(P[0][0], 100, range=(0,1))
    plt.figure(1)
    plt.title("Milestone results - Positive pairs, BLEU, ALECS (n = 101895)")
    plt.xlabel("BLEU score (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(P[1][0], 2000, range=(0,1))
    
    plt.figure(2)
    plt.title("Milestone results - Positive pairs, BERTScore, Webis (n = 4067)")
    plt.xlabel("BERTScore (\u0394 = 0.01)")
    plt.ylabel("Frequencies")
    plt.hist(P[0][1], 100, range=(0,1))
    
    plt.figure(3)
    plt.title("Milestone results - Positive pairs, BERTScore, ALECS (n = 101895)")
    plt.xlabel("BERTScore (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(P[1][1], 2000, range=(0,1))
    
    plt.figure(4)
    plt.title("Milestone results - Positive pairs, SE-F1, Webis (n = 4067)")
    plt.xlabel("SE-F1 score (\u0394 = 0.01)")
    plt.ylabel("Frequencies")
    plt.hist(P[0][2], 100, range=(0,1))
    
    plt.figure(5)
    plt.title("Milestone results - Positive pairs, SE-F1, ALECS (n = 101895)")
    plt.xlabel("SE-F1 score (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(P[1][2], 2000, range=(0,1))
    
    
    plt.figure(6)
    plt.title("Milestone results - Negative pairs, BLEU, ALECS* (n = 100000)")
    plt.xlabel("BLEU score (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(N[0], 2000, range=(0,1))
    
    plt.figure(7)
    plt.title("Milestone results - Negative pairs, BERTScore, ALECS* (n = 100000)")
    plt.xlabel("BERTScore (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(N[1], 2000, range=(0,1))
    
    plt.figure(8)
    plt.title("Milestone results - Negative pairs, SE-F1, ALECS* (n = 100000)")
    plt.xlabel("SE-F1 Score (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(N[2], 2000, range=(0,1))
    
    plt.show()