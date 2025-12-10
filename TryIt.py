#A simple suite to try the tuned, calibrated model!


#####
MODEL = "Model/sef1-all-mpnet-base-v2/curr"
CALIBRATIONMAP_PATH = "Model/CalibrationMap.cpkl"
#####

if __name__ == "__main__":
    print("Takes a while to load...")
    import torch
    from Predictors.BLEU import BLEU
    from Predictors.BERTScore import BERTScore
    from Predictors.SE_F1 import SE_F1, CalibratedSE_F1
    from Utils.Utils import Recover_Object
    torch.set_default_device("cpu")
    print("done!")
    
    BL = BLEU()
    BS = BERTScore()
    SF = SE_F1()
    CSF = CalibratedSE_F1(MODEL, Recover_Object(CALIBRATIONMAP_PATH))
    print("Predictors loaded.")
    p1 = input("Enter the first paragraph: ")
    p2 = input("Enter the first paragraph: ")
    print("BLEU: ", BL.score([p1], [p2]))
    print("BERTScore: ", BS.score([p1], [p2]))
    print("SE-F1 (base)", SF.score([p1], [p2]))
    print("Trained Predictor (calibrated SE-F1)", CSF.score([p1], [p2]))