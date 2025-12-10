# -*- coding:utf-8 -*-
import torch
import gc
import time

import datasets
from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainingArguments, \
                                    SentenceTransformerTrainer
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.sampler import DefaultBatchSampler
from Predictors.SE_F1 import SE_F1
from Utils.Dataset_Gen import Train_Datasets_Combine, Concatenate_Datasets
#torch.autograd.set_detect_anomaly(True)
#####
DEF_DTYPE = torch.float16
DEF_TORCH_DEVICE = "cuda" #modify if run from non-cuda environment
DEF_MODEL = "all-mpnet-base-v2"
MOD_PATH = "Model/sef1-all-mpnet-base-v2/curr/"
MOD_SUB_PATH = "Model/sef1-all-mpnet-base-v2/"
#####
torch.set_default_device(DEF_TORCH_DEVICE)

class SEF1MSE(torch.nn.Module):
    ##Loss function via SE-F1 procedure
    ##    ... directly calculate MSE from SE_F1
    ##
    ##INPUT FORMAT:
    ##    texts: "two columns" containing two paragraphs (positive or negative)
    ##    labels: List of 1 (positive) or 0 (negative) pairs
    ##ONLY SUPPORTS MODELS WITH TOKENIZERS WITH MARKED "bos_token" AND "eos_token" SEPATING PARAGRAPHS
    ##    (by the structure of the dataset used, each paragraph)
    def __init__(self, model, loss = torch.nn.MSELoss()):
        super().__init__()
        self.model = model
        self.sef1 = SE_F1(self.model)
        self.loss = loss #actual parent loss function 
    def forward(self, sentence_features, labels) -> float:
        #batch MSE via SE-F1 scores
        
        #decode sentence_features into strings 
        #    (there probably is a better way than decoding just for it to be re-encoded)
        batch_pars = [x["input_ids"] for x in sentence_features]
        assert len(batch_pars) == 2 #should be 2-long for the 2 columns
        tokenizer = self.model.tokenizer
        bos = tokenizer.special_tokens_map["bos_token"]
        eos = tokenizer.special_tokens_map["eos_token"]
        par1 = [] #list of first paragraphs, eos and bos tokens removed
        par2 = [] #list of second paragraphs, ``
        for i in range(len(batch_pars[0])):
            x,y = tokenizer.decode(batch_pars[0][i]), tokenizer.decode(batch_pars[1][i])
            par1.append(x[x.find(bos) + len(bos) : x.find(eos)])
            par2.append(y[y.find(bos) + len(bos) : y.find(eos)])
        """
        print(batch_pars)
        print("=======")
        print(par1)
        print(par2)
        print("=======")
        print(labels)
        
        print("check res running")
        """
        
        res = self.sef1._torch_score(par1, par2)
        """
        print("heh")
        
        print("=======")
        print(res, res.requires_grad)
        """
        loss = self.loss(res, labels.float().view(-1))
        return loss
        
def DATASET_INIT():
    #RUN ONCE!!!!
    #initializes the datasets.
    from Utils.Dataset_Gen import Load_All_Pairs_From_Pickle_Dir, Dataset_Creation
    
    D = Load_All_Pairs_From_Pickle_Dir("Datasets/Paraphrases/")
    #print(D["w+"][0])
    Dataset_Creation(D, "Datasets/Train", "Datasets/CV", "Datasets/Test")

if __name__ == "__main__":
    """#Initialize dataset
    DATASET_INIT()
    """
    #vvv
    TRAIN_DIR, CV_DIR, TEST_DIR = "Datasets/Train", "Datasets/CV", "Datasets/Test"
    #^^^
    
    train = Train_Datasets_Combine(TRAIN_DIR, keys = ["w+", "w-", "a+", "a-"], ratios=[0.4,0.4,0.1,0.1], full = False)
    cv = Concatenate_Datasets(CV_DIR)
    """
    #hide the tests
    test = Concatenate_Datasets(TEST_DIR)
    """
    print("##DATASETS LOADED##")
    print(f"Train dataset looks like: {train}")
    
    """
    i = 0 #train inspection
    print("first entry (" + str(train["label"][i]) + "): " + repr(train["paragraphs1"][i]) + "\n====\n" + repr(train["paragraphs2"][i]))
    for i in range(1,10):
        print(str(train["label"][i]) + "; " + repr(train["paragraphs1"][i]) + "\n====\n" + repr(train["paragraphs2"][i]))
    
    """
    if input(f"type 't' to reset the model stored in MOD_PATH ({MOD_PATH}) ").lower() == "t":
        print("guarded")
        assert False
        model = SentenceTransformer(DEF_MODEL, device = DEF_TORCH_DEVICE)
        print("##MODEL LOADED##")
    else:
        model = SentenceTransformer(MOD_PATH, device = DEF_TORCH_DEVICE)
        print("##MODEL LOADED##")
    
    import sys
    
    orig_stdout = sys.stdout
    
    f = open("log.txt", "a")
    
    sys.stdout = f
    print("\n\n========\n")
    first = True
    train_length = len(train)
    SUBEPOCH_SIZE = 4
    for ep in range(4):
        #shuffle each epoch
        print(f"####EPOCH {ep}")
        train = train.shuffle()
        for c_se in range(((train_length - 1) // SUBEPOCH_SIZE) + 1):
            print(f"###SUB_EPOCH {c_se} / {((train_length - 1) // SUBEPOCH_SIZE)} ({int(time.time())})")
            if not first:
                model = SentenceTransformer(MOD_PATH, device = DEF_TORCH_DEVICE)
            curr_train = train.select(range(c_se * SUBEPOCH_SIZE, min(train_length, (c_se + 1) * SUBEPOCH_SIZE)))
            loss = SEF1MSE(model)
            #loss = CosineSimilarityLoss(model)
            args = SentenceTransformerTrainingArguments(
                output_dir=MOD_SUB_PATH,
                num_train_epochs=1,
                per_device_train_batch_size=4,
                learning_rate=5e-5,
                lr_scheduler_type="constant",
                fp16=True,
                weight_decay=0.0, 
                max_grad_norm=2.0,
                optim='sgd',
                #bf16=True,
                save_strategy="no", #manual save
                logging_first_step = True,
                logging_steps=1,
                logging_strategy="steps",
                dataloader_pin_memory=False,
            )
            
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=curr_train,
                loss=loss,
            )
            trainer.train()
            model.save_pretrained(MOD_PATH)
            
            #cleanup
            first = False
            del model, curr_train, trainer, loss
            gc.collect()
            if DEF_TORCH_DEVICE == "cuda":
                torch.cuda.empty_cache()
            if DEF_TORCH_DEVICE == "mps":
                torch.mps.empty_cache()
            