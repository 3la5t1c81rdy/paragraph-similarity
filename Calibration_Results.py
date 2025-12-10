# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from Predictors.SE_F1 import CalibratedSE_F1
from Utils.Utils import Save_Object, Recover_Object, MSE

#####
MODELDIR = "Model/"
MODEL = "Model/sef1-all-mpnet-base-v2/curr"
CALIBRATIONMAP_PATH = "Model/CalibrationMap.cpkl"
#####

if __name__ == "__main__":
    """
    #model = SentenceTransformer(MODEL) #load model
    model = None #save resources!
    RM = Recover_Object(CALIBRATIONMAP_PATH)
    Cali_SF = CalibratedSE_F1(model, RM, disable_score = True)
    
    #"re-grade" each of train, CV, test
    trwp, cvwp, cvap, cvan = Recover_Object(os.path.join(MODELDIR, "trainRes.cpkl"))
    trwn, trap, tran, cvwn = Recover_Object(os.path.join(MODELDIR, "trainRes2.cpkl"))
    
    c_trwp = Cali_SF.maps(trwp)
    c_trwn = Cali_SF.maps(trwn)
    c_trap = Cali_SF.maps(trap)
    c_tran = Cali_SF.maps(tran)
    c_cvwp = Cali_SF.maps(cvwp)
    c_cvwn = Cali_SF.maps(cvwn)
    c_cvap = Cali_SF.maps(cvap)
    c_cvan = Cali_SF.maps(cvan)
    
    Save_Object([c_trwp, c_trwn, c_trap, c_tran, c_cvwp, c_cvwn, \
                 c_cvap, c_cvan], os.path.join(MODELDIR, "calibrationRes.cpkl"))
    """
    
    """#shivers v2. Test set on Calibration
    testwp, testwn, testap, testan = Recover_Object(os.path.join(MODELDIR, "trainResTest.cpkl"))
    c_testwp = Cali_SF.maps(testwp)
    c_testwn = Cali_SF.maps(testwn)
    c_testap = Cali_SF.maps(testap)
    c_testan = Cali_SF.maps(testan)
    
    Save_Object([c_testwp, c_testwn, c_testap, c_testan], os.path.join(MODELDIR, "calibrationResTest.cpkl"))
    
    """
    #^^^ Above has been run
    
    #re-collect the results for graphing
    c_trwp, c_trwn, c_trap, c_tran, c_cvwp, c_cvwn, \
        c_cvap, c_cvan = Recover_Object(os.path.join(MODELDIR, "calibrationRes.cpkl"))
    c_testwp, c_testwn, c_testap, c_testan = Recover_Object(os.path.join(MODELDIR, "calibrationResTest.cpkl"))
    
    
    #(now compute loss for calibrated model)
    #Losses calculated separately.
    print("Calibrated Training loss (Webis): " + str(MSE(c_trwp + c_trwn, \
                                  [1 for _ in range(len(c_trwp))] + [0 for _ in range(len(c_trwn))], \
                                  weight=[len(c_trwn)/len(c_trwp) for _ in range(len(c_trwp))] + [1 for _ in range(len(c_trwn))])))
    print("Calibrated Training loss (ALECS): " + str(MSE(c_trap + c_tran, \
                                  [1 for _ in range(len(c_trap))] + [0 for _ in range(len(c_tran))])))
    
    print("Calibrated CV loss (Webis): " + str(MSE(c_cvwp + c_cvwn, \
                                  [1 for _ in range(len(c_cvwp))] + [0 for _ in range(len(c_cvwn))], \
                                  weight=[len(c_cvwn)/len(c_cvwp) for _ in range(len(c_cvwp))] + [1 for _ in range(len(c_cvwn))])))
    print("Calibrated CV loss (ALECS): " + str(MSE(c_cvap + c_cvan, \
                                  [1 for _ in range(len(c_cvap))] + [0 for _ in range(len(c_cvan))])))
    
    print("Calibrated TEST loss (Webis): " + str(MSE(c_testwp + c_testwn, \
                                  [1 for _ in range(len(c_testwp))] + [0 for _ in range(len(c_testwn))], \
                                  weight=[len(c_testwn)/len(c_testwp) for _ in range(len(c_testwp))] + [1 for _ in range(len(c_testwn))])))
    print("Calibrated TEST loss (ALECS): " + str(MSE(c_testap + c_testan, \
                                  [1 for _ in range(len(c_testap))] + [0 for _ in range(len(c_testan))])))
    
    
    #... mostly copied from Train_Results
    rcParams["figure.figsize"] = [12.8, 9.6]
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("Train Webis")
    ax1.set_xlabel("Calibrated, Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(c_trwn, 100, range=(0,1), label = "- (n = 71326, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(c_trwp, 100, range=(0,1), label = "+ (n = 2846, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(1)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("Train ALECS")
    ax1.set_xlabel("Calibrated, Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(c_tran, 100, range=(0,1), label = "- (n = 71326, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(c_trap, 100, range=(0,1), label = "+ (n = 71326, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(2)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("CV Webis")
    ax1.set_xlabel("Calibrated, Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(c_cvwn, 100, range=(0,1), label = "- (n = 15284, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(c_cvwp, 100, range=(0,1), label = "+ (n = 610, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(3)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("CV ALECS")
    ax1.set_xlabel("Calibrated, Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(c_cvan, 100, range=(0,1), label = "- (n = 15284, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(c_cvap, 100, range=(0,1), label = "+ (n = 15284, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(4)
    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("TEST Webis")
    ax1.set_xlabel("Calibrated, Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(c_testwn, 100, range=(0,1), label = "- (n = 15285, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(c_testwp, 100, range=(0,1), label = "+ (n = 611, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(5)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title("TEST ALECS")
    ax1.set_xlabel("Calibrated, Tuned SE-F1 score")
    ax1.set_ylabel("Frequencies (-)", color="red")
    ax1.tick_params(axis="y", color="red", labelcolor="red")
    h1 = ax1.hist(c_testan, 100, range=(0,1), label = "- (n = 15285, \u0394 = 0.01)", color="red", alpha=0.5)
    ax2.set_ylabel("Frequencies (+)", color="blue")
    ax2.tick_params(axis="y", color="blue", labelcolor="blue")
    h2 = ax2.hist(c_testap, 100, range=(0,1), label = "+ (n = 15285, \u0394 = 0.01)", color="blue", alpha=0.5)
    fig.legend()
    plt.figure(6)
    
    plt.show()