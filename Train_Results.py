# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import datasets
import numpy as np
from Predictors.SE_F1 import SE_F1
from sentence_transformers import SentenceTransformer
from Utils.Utils import Save_Object, Recover_Object, MSE

#####
MODELDIR = "Model/"
MODEL = "Model/sef1-all-mpnet-base-v2/curr"
TRAIN_LOG = "log.txt"
#####


if __name__ == "__main__":
    #actual train-time loss over SGD iterations
    #interpret the log file
    L = []
    with open(TRAIN_LOG, "r") as f:
        sep_ct = 0
        batch_size_map = {1:0.5, 2:1, 3:1}
        currline = ""
        x = 0
        while len(currline := f.readline()) != 0:
            if "========" in currline:
                sep_ct += 1
            if "'train_loss': " in currline:
                currline = currline[currline.find("'train_loss': ") + len("'train_loss': "):]
                currline = currline[:currline.find(",")]
                L.append((x + batch_size_map[sep_ct], float(currline)))
                x += batch_size_map[sep_ct]
    sub = L
    max_bn = 0
    L = []
    for i in range(0, len(sub), 8):
        s, t = [], 0
        for j in range(i, min(i + 8, len(sub))):
            t = sub[j][0]  #most recent batch #
            if j == 0:
                s.append((sub[j][0], sub[j][1])) #(batch size, batch loss)
            else:
                s.append((sub[j][0] - sub[j-1][0], sub[j][1]))
        max_bn = t #save the max batch number
        a,b = zip(*s)
        L.append((t, sum([a[i] * b[i] for i in range(len(a))]) / sum(a))) #{ MSE -> SSE } up to 8 times, then divide by the sum of batch sizes
    plt.title("Training Loss Over Time")
    plt.xlabel("Batch (k = 4)")
    plt.ylabel("MSE")
    plt.plot(*zip(*sub), label="MSE")
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=max_bn)
    plt.legend()
    plt.show()
    plt.title("Average Training Loss of 8 Consecutive Batches")
    plt.xlabel("Batch (k = 4)")
    plt.ylabel("MSE")
    m, b = np.polyfit(*zip(*L), 1)
    plt.plot(*zip(*L), label="Average MSE")
    plt.plot([0, max_bn], [b, m * max_bn + b], linestyle="dashed", label="Line of best fit")
    plt.ylim(bottom=0)
    plt.xlim(left=0, right=max_bn)
    plt.legend()
    plt.show()
    
    """
    dwp_train = datasets.Dataset.load_from_disk("Datasets/Train/w+")
    dwp_cv = datasets.Dataset.load_from_disk("Datasets/CV/w+")
    dap_cv = datasets.Dataset.load_from_disk("Datasets/CV/a+")
    dan_cv = datasets.Dataset.load_from_disk("Datasets/CV/a-")
    print(len(dwp_train), len(dwp_cv))
    """
    """
    model = SentenceTransformer(MODEL)
    SF = SE_F1(model)
    """
    """
    trwp = SF.score(dwp_train["paragraphs1"][:], dwp_train["paragraphs2"][:], info=True)
    cvwp = SF.score(dwp_cv["paragraphs1"][:], dwp_cv["paragraphs2"][:], info=True)
    cvap = SF.score(dap_cv["paragraphs1"][:], dap_cv["paragraphs2"][:], info=True)
    cvan = SF.score(dan_cv["paragraphs1"][:], dan_cv["paragraphs2"][:], info=True)
    
    Save_Object([trwp, cvwp, cvap, cvan], "trainRes.cpkl")
    """
    """
    dwn_train = datasets.Dataset.load_from_disk("Datasets/Train/w-")
    dap_train = datasets.Dataset.load_from_disk("Datasets/Train/a+")
    dan_train = datasets.Dataset.load_from_disk("Datasets/Train/a-")
    dwn_cv = datasets.Dataset.load_from_disk("Datasets/CV/w-")
    
    print("trwn")
    trwn = SF.score(dwn_train["paragraphs1"][:], dwn_train["paragraphs2"][:], info=True)
    print("trap")
    trap = SF.score(dap_train["paragraphs1"][:], dap_train["paragraphs2"][:], info=True)
    print("tran")
    tran = SF.score(dan_train["paragraphs1"][:], dan_train["paragraphs2"][:], info=True)
    print("cvwn")
    cvwn = SF.score(dwn_cv["paragraphs1"][:], dwn_cv["paragraphs2"][:], info=True)
    
    Save_Object([trwn, trap, tran, cvwn], "trainRes2.cpkl")
    """
    
    """#shivers. This is the test.
    dwp_test = datasets.Dataset.load_from_disk("Datasets/Test/w+")
    dwn_test = datasets.Dataset.load_from_disk("Datasets/Test/w-")
    dap_test = datasets.Dataset.load_from_disk("Datasets/Test/a+")
    dan_test = datasets.Dataset.load_from_disk("Datasets/Test/a-")
    
    print("testwp")
    testwp = SF.score(dwp_test["paragraphs1"][:], dwp_test["paragraphs2"][:], info=True)
    print("testwn")
    testwn = SF.score(dwn_test["paragraphs1"][:], dwn_test["paragraphs2"][:], info=True)
    print("testap")
    testap = SF.score(dap_test["paragraphs1"][:], dap_test["paragraphs2"][:], info=True)
    print("testan")
    testan = SF.score(dan_test["paragraphs1"][:], dan_test["paragraphs2"][:], info=True)
    
    Save_Object([testwp, testwn, testap, testan], "trainResTest.cpkl")
    """
    
    trwp, cvwp, cvap, cvan = Recover_Object(os.path.join(MODELDIR, "trainRes.cpkl"))
    trwn, trap, tran, cvwn = Recover_Object(os.path.join(MODELDIR, "trainRes2.cpkl"))
    testwp, testwn, testap, testan = Recover_Object(os.path.join(MODELDIR, "trainResTest.cpkl"))
    
    #(reconstruct losses for convenience)
    #Losses calculated separately.
    print("Training loss (Webis): " + str(MSE(trwp + trwn, \
                                  [1 for _ in range(len(trwp))] + [0 for _ in range(len(trwn))], \
                                  weight=[len(trwn)/len(trwp) for _ in range(len(trwp))] + [1 for _ in range(len(trwn))])))
    print("Training loss (ALECS): " + str(MSE(trap + tran, \
                                  [1 for _ in range(len(trap))] + [0 for _ in range(len(tran))])))
    
    print("CV loss (Webis): " + str(MSE(cvwp + cvwn, \
                                  [1 for _ in range(len(cvwp))] + [0 for _ in range(len(cvwn))], \
                                  weight=[len(cvwn)/len(cvwp) for _ in range(len(cvwp))] + [1 for _ in range(len(cvwn))])))
    print("CV loss (ALECS): " + str(MSE(cvap + cvan, \
                                  [1 for _ in range(len(cvap))] + [0 for _ in range(len(cvan))])))
    
    print("TEST loss (Webis): " + str(MSE(testwp + testwn, \
                                  [1 for _ in range(len(testwp))] + [0 for _ in range(len(testwn))], \
                                  weight=[len(testwn)/len(testwp) for _ in range(len(testwp))] + [1 for _ in range(len(testwn))])))
    print("TEST loss (ALECS): " + str(MSE(testap + testan, \
                                  [1 for _ in range(len(testap))] + [0 for _ in range(len(testan))])))
    
    rcParams["figure.figsize"] = [12.8, 9.6]
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("Train Webis")
    ax1.set_xlabel("Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(trwn, 100, range=(0,1), label = "- (n = 71326, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(trwp, 100, range=(0,1), label = "+ (n = 2846, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(1)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("Train ALECS")
    ax1.set_xlabel("Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(tran, 1000, range=(0,1), label = "- (n = 71326, \u0394 = 0.001)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(trap, 1000, range=(0,1), label = "+ (n = 71326, \u0394 = 0.001)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(2)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("CV Webis")
    ax1.set_xlabel("Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(cvwn, 100, range=(0,1), label = "- (n = 15284, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(cvwp, 100, range=(0,1), label = "+ (n = 610, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(3)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("CV ALECS")
    ax1.set_xlabel("Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(cvan, 500, range=(0,1), label = "- (n = 15284, \u0394 = 0.002)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(cvap, 500, range=(0,1), label = "+ (n = 15284, \u0394 = 0.002)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(4)
    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("TEST Webis")
    ax1.set_xlabel("Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(testwn, 100, range=(0,1), label = "- (n = 15285, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(testwp, 100, range=(0,1), label = "+ (n = 611, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(5)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("TEST ALECS")
    ax1.set_xlabel("Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(testan, 500, range=(0,1), label = "- (n = 15285, \u0394 = 0.002)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(testap, 500, range=(0,1), label = "+ (n = 15285, \u0394 = 0.002)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(6)
    
    """
    plt.title("Train Webis + (n = 2846)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.01)")
    plt.ylabel("Frequencies")
    plt.hist(trwp, 100, range=(0,1))
    plt.figure(1)
    plt.title("CV Webis + (n = 610)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.01)")
    plt.ylabel("Frequencies")
    plt.hist(cvwp, 100, range=(0,1))
    plt.figure(2)
    plt.title("CV ALECS + (n = 15284)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.002)")
    plt.ylabel("Frequencies")
    plt.hist(cvap, 500, range=(0,1))
    plt.figure(3)
    plt.title("CV ALECS - (n = 15284)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.002)")
    plt.ylabel("Frequencies")
    plt.hist(cvan, 500, range=(0,1))
    
    
    plt.figure(4)
    plt.title("Train Webis - (n = 71326)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.001)")
    plt.ylabel("Frequencies")
    plt.hist(trwn, 1000, range=(0,1))
    plt.figure(5)
    plt.title("Train ALECS + (n = 71326)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.001)")
    plt.ylabel("Frequencies")
    plt.hist(trap, 1000, range=(0,1))
    plt.figure(6)
    plt.title("Train ALECS - (n = 71326)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.001)")
    plt.ylabel("Frequencies")
    plt.hist(tran, 1000, range=(0,1))
    plt.figure(7)
    plt.title("CV Webis - (n = 15284)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.002)")
    plt.ylabel("Frequencies")
    plt.hist(cvwn, 500, range=(0,1))
    
    
    plt.figure(8)
    plt.title("Test Webis + (n = 611)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.01)")
    plt.ylabel("Frequencies")
    plt.hist(testwp, 100, range=(0,1))
    plt.figure(9)
    plt.title("Test Webis - (n = 15285)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.002)")
    plt.ylabel("Frequencies")
    plt.hist(testwn, 500, range=(0,1))
    plt.figure(10)
    plt.title("Test ALECS + (n = 15285)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.002)")
    plt.ylabel("Frequencies")
    plt.hist(testap, 500, range=(0,1))
    plt.figure(11)
    plt.title("Test ALECS - (n = 15285)")
    plt.xlabel("Mod SE-F1 score (\u0394 = 0.002)")
    plt.ylabel("Frequencies")
    plt.hist(testan, 500, range=(0,1))
    """
    
    plt.show()