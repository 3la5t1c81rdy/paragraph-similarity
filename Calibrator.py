# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
import datasets
from Utils.Utils import Save_Object, Recover_Object, Fit_Calibration

#####
MODELDIR = "Model/"
SAVE_MAPPING = False
#####

#Crucial to only calibrate on the train set. Let's see how it "improves" the results.
if __name__ == "__main__":
    #To induce "fairness" between groups (datasets), I will MULTIPLY the [w+] results (2846) to match the count of
    #    [a+], [a-], and [w-] results (71326 each) ... (randomly picking the remainders)
    #from each of w-, a+, a- datasets.
    #I exploit the fact that patch calibration only requires the RESULTS (return values/predictions) of the model
    #as well as the true labels, so this calibration can be done completely post-hoc, with zero knowledge
    #    of the original inputs (features of the sample).
    #
    #Because the output is continuous, the calibration will be "approximated" by 
    #BINNING/"rounding" the results (as implemented in Fit_Calibration)
    #
    #I will calibrate with alpha = 1e-7, and with 100 bins
    #    (that gives 0.0, 0.01, 0.02, ..., 0.99, 1 as "roundoffs" so actually "101 actual bins")
    
    trwp, cvwp, cvap, cvan = Recover_Object(os.path.join(MODELDIR, "trainRes.cpkl"))
    trwn, trap, tran, cvwn = Recover_Object(os.path.join(MODELDIR, "trainRes2.cpkl"))
    
    qr = None
    print(len(trwp), len(trwn), len(trap), len(tran)) #2846 71326 71326 71326
    assert len(trwn) == len(trap) and len(trap) == len(tran)
    print(qr := (len(trwn) // len(trwp), len(trwn) % len(trwp))) #(25, 176)
    
    mult_trwp = trwp * qr[0] + random.sample(trwp, qr[1])
    print(len(mult_trwp)) #71326 == len(trwn) == len(trap) == len(tran)
    aggr_preds = mult_trwp + trap + trwn + tran
    aggr_labels = [1 for _ in range(2 * len(trwn))] + [0 for _ in range(2 * len(trwn))]
    print(len(aggr_preds), len(aggr_labels)) #285304 285304
    print(aggr_labels[142651], aggr_labels[142652]) #1 0
    
    #Calibrate!
    RM = Fit_Calibration(aggr_preds, aggr_labels, bins=100) #takes ~1-2 minutes
    
    
    print(RM.mapping)
    if SAVE_MAPPING:
        Save_Object(RM, os.path.join(MODELDIR, "CalibrationMap.cpkl")) #store away!
    
    ##Test the mapping! (on CV of course)
    s_cvwp = random.sample(cvwp, 10)
    s_cvwn = random.sample(cvwn, 10)
    s_cvap = random.sample(cvap, 10)
    s_cvan = random.sample(cvan, 10)
    print("10 random CV w+ samples:", [RM.map(x) for x in s_cvwp])
    print("10 random CV w- samples:", [RM.map(x) for x in s_cvwn])
    print("10 random CV a+ samples:", [RM.map(x) for x in s_cvap])
    print("10 random CV a- samples:", [RM.map(x) for x in s_cvan])
    print("Pretty robust (for the most part)!")