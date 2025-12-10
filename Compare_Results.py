# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Utils.Utils import Recover_Object, Pearson_Corr, MSE

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
    #Negatives only contain ALECS-sourced pairs. (OUTDATED)
    # => modified to only get the negatives if marked as ALECS
    #Returns a list of results in order of BLEU, BERTSCORE, SE_F1
    Aggr = ([[],[],[]], [[],[],[]])
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
        if t.find(PREF_W) == 0:
            i = 0
        elif t.find(PREF_A) == 0:
            i = 1
        else:
            #print(fn, "??")
            #ignore Webis
            continue
        
        curr = Recover_Object(os.path.join(RES_NEG_DIR, fn))
        Aggr[i][j] += curr
        
    
    return Aggr

if __name__ == "__main__":
    #draw the Results from the stored files!
    P = Aggr_Positives()
    N = Aggr_Negatives()
    print(len(P[0][0]))
    print(len(P[1][0]))
    #print(N[0])
    """
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
    plt.hist(N[0][0], 2000, range=(0,1))
    
    plt.figure(7)
    plt.title("Milestone results - Negative pairs, BERTScore, ALECS* (n = 100000)")
    plt.xlabel("BERTScore (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(N[0][1], 2000, range=(0,1))
    
    plt.figure(8)
    plt.title("Milestone results - Negative pairs, SE-F1, ALECS* (n = 100000)")
    plt.xlabel("SE-F1 Score (\u0394 = 0.0005)")
    plt.ylabel("Frequencies")
    plt.hist(N[0][2], 2000, range=(0,1))
    """
    #correlation in Webis and ALECS datasets (+)
    print("Webis", Pearson_Corr(P[0][0], P[0][1], P[0][2]))
    print("ALECS", Pearson_Corr(P[1][0], P[1][1], P[1][2]))
    
    print("SE-F1 Loss (Webis): " + str(MSE(P[0][2] + N[0][2], \
                                  [1 for _ in range(len(P[0][2]))] + [0 for _ in range(len(N[0][2]))], \
                                  weight=[len(N[0][2])/len(P[0][2]) for _ in range(len(P[0][2]))] + [1 for _ in range(len(N[0][2]))])))
    print("SE-F1 Loss (ALECS): " + str(MSE(P[1][2] + N[1][2], \
                                  [1 for _ in range(len(P[1][2]))] + [0 for _ in range(len(N[1][2]))])))
    
    #graph
    rcParams["figure.figsize"] = [12.8, 9.6]
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("BLEU on Webis")
    ax1.set_xlabel("BLEU score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(N[0][0], 100, range=(0,1), label = "- (n = 101895, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(P[0][0], 100, range=(0,1), label = "+ (n = 4067, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    
    plt.figure(1)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("BLEU on ALECS")
    ax1.set_xlabel("BLEU score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(N[1][0], 1000, range=(0,1), label = "- (n = 100000, \u0394 = 0.001)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(P[1][0], 1000, range=(0,1), label = "+ (n = 101895, \u0394 = 0.001)", color="blue", alpha=0.5)
    
    fig.legend()
    
    
    plt.figure(2)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("BERTScore on Webis")
    ax1.set_xlabel("BERTScore")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(N[0][1], 100, range=(0,1), label = "- (n = 101895, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(P[0][1], 100, range=(0,1), label = "+ (n = 4067, \u0394 = 0.01)", color="blue", alpha=0.5)
    
    fig.legend()
    
    
    plt.figure(3)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("BERTScore on ALECS")
    ax1.set_xlabel("BERTScore")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(N[1][1], 1000, range=(0,1), label = "- (n = 100000, \u0394 = 0.001)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(P[1][1], 1000, range=(0,1), label = "+ (n = 101895, \u0394 = 0.001)", color="blue", alpha=0.5)
    
    fig.legend()
    
    
    plt.figure(4)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("SE-F1 on Webis")
    ax1.set_xlabel("SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(N[0][2], 100, range=(0,1), label = "- (n = 101895, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(P[0][2], 100, range=(0,1), label = "+ (n = 4067, \u0394 = 0.01)", color="blue", alpha=0.5)
    
    fig.legend()
    
    
    plt.figure(5)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("SE-F1 on ALECS")
    ax1.set_xlabel("SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(N[1][2], 1000, range=(0,1), label = "- (n = 100000, \u0394 = 0.001)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(P[1][2], 1000, range=(0,1), label = "+ (n = 101895, \u0394 = 0.001)", color="blue", alpha=0.5)
    
    fig.legend()
    
    plt.show()