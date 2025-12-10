# -*- coding:utf-8 -*-
import os

import datasets
from datasets import Dataset, interleave_datasets, concatenate_datasets
from Utils import Recover_Object, Save_Object

#####
W_POSITIVE_PREFIX = "W-pairs-"
W_NEGATIVE_PREFIX = "W-negative-pairs-"
A_POSITIVE_PREFIX = "A-pairs-"
A_NEGATIVE_PREFIX = "A-negative-pairs-"
PICKLE_SUFFIX = ".cpkl" #"<PREFIX>-<#p><PICKLE_SUFFIX>" is the file format
#####

""" #playground code
for _ in range(10):
    d1 = Dataset.from_dict({"u":[1 for _ in range(1000)]})
    d2 = Dataset.from_dict({"u":[2 for _ in range(1000)]})
    d3 = Dataset.from_dict({"u":[3 for _ in range(1000)]})
    d4 = Dataset.from_dict({"u":[5 for _ in range(99)] + [4 for _ in range(500)]})
    #yields different result from
    #    d4 = Dataset.from_dict({"u":[4 for _ in range(500)] + [5 for _ in range(99)]})
    #=> shuffle it!
    aggr = interleave_datasets([d1, d2, d3, d4], stopping_strategy="all_exhausted")
    cts = [[None for x in aggr["u"] if x == i] for i in range(1,6)]
    cts = [len(cts[i]) for i in range(len(cts))]
    print(cts)

assert False
"""

def Load_All_Pairs_From_Pickle_Dir(pkl_dir_path) \
     -> dict["w+":list[(str,str)], "w-":list[(str,str)], "a+":list[(str,str)], "a-":list[(str,str)]]:
    #collects all positive and negative pairs as saved in pkl_dir_path directory and assorts them
    #    all into a gigantic dictionary.
    #file "identification" is done purely by looking at the file prefix (and suffix)
    #only gathers the actual pairs ()
    if (not os.path.isdir(pkl_dir_path)) or \
        (not os.path.isfile(os.path.join(pkl_dir_path, W_POSITIVE_PREFIX+"0"+PICKLE_SUFFIX))) or \
        (not os.path.isfile(os.path.join(pkl_dir_path, W_NEGATIVE_PREFIX+"0"+PICKLE_SUFFIX))) or \
        (not os.path.isfile(os.path.join(pkl_dir_path, A_POSITIVE_PREFIX+"0"+PICKLE_SUFFIX))) or \
        (not os.path.isfile(os.path.join(pkl_dir_path, A_NEGATIVE_PREFIX+"0"+PICKLE_SUFFIX))):
        raise FileNotFoundError(f"{pkl_dir_path} is an invalid pickle directory.")
    rd = {"w+":[], "w-":[], "a+":[], "a-":[]}
    c = None
    l = os.listdir(pkl_dir_path)
    for f in l:
        #filenames may be shuffled
        if f.find(W_POSITIVE_PREFIX) == 0 and f.find(PICKLE_SUFFIX) == len(f) - len(PICKLE_SUFFIX):
            c = "w+"
        elif f.find(W_NEGATIVE_PREFIX) == 0 and f.find(PICKLE_SUFFIX) == len(f) - len(PICKLE_SUFFIX):
            c = "w-"
        elif f.find(A_POSITIVE_PREFIX) == 0 and f.find(PICKLE_SUFFIX) == len(f) - len(PICKLE_SUFFIX):
            c = "a+"
        elif f.find(A_NEGATIVE_PREFIX) == 0 and f.find(PICKLE_SUFFIX) == len(f) - len(PICKLE_SUFFIX):
            c = "a-"
        else:
            #invalid file; ignore
            continue
        f = os.path.join(pkl_dir_path, f)
        if not os.path.isfile(f):
            print("This shouldn't occur")
            continue
        rd[c] += [(a, b) for (a, b, _, _) in Recover_Object(f)] #drop filepaths
    return rd

def Dataset_Creation(pairs_dict:dict["w+","w-","a+","a-"], train_path, cv_path, test_path):
    #takes the dataset pairs as returned by Load_All_Pairs_From_Pickle_Dir
    #    and generates train, cv, test splits (0.7, 0.15, 0.15)
    #the dataset (and each splits) are FRAGMENTED into the four sub-directories and saved
    #    (into respective disk locations)
    #each datasets consist solely of columns "paragraphs1", "paragraphs2", "label"
    #    (+ datasets have labels [1,1,...,1] and - datasets have labels [0,0,...,0])
    for cdn in ["w+", "w-", "a+", "a-"]:
        if "+" in cdn:
            label = [1 for _ in range(len(pairs_dict[cdn]))]
        else:
            label = [0 for _ in range(len(pairs_dict[cdn]))]
        par1 = [x for (x, y) in pairs_dict[cdn]]
        par2 = [y for (x, y) in pairs_dict[cdn]]
        cd = Dataset.from_dict({"paragraphs1":par1, "paragraphs2":par2, "label":label})
        cd = cd.train_test_split(train_size=0.7, shuffle=True) #70-30
        train = cd["train"]
        test = cd["test"]
        test = test.train_test_split(test_size=0.5, shuffle=True)#(70-)30/2-30/2
        cv = test["train"]
        test = test["test"]
        #separation!
        train.save_to_disk(os.path.join(train_path, cdn))
        test.save_to_disk(os.path.join(test_path, cdn))
        cv.save_to_disk(os.path.join(cv_path, cdn))

def Train_Datasets_Combine(train_path, keys = ["w+", "w-", "a+", "a-"], ratios=[0.25, 0.25, 0.25, 0.25], full = True) -> Dataset:
    #Interleaves the train dataset (each of the four pos/neg datasets) into an aggregate
    #    that has "equal proportion of each datasets" (despite w+ being very small)
    #    ... suitable for training
    #train_path should be as used in Dataset_Creation (i.e. the "parent" directory path)
    #customize the split ratios via [ratios] parameter
    
    if (not os.path.isdir(os.path.join(train_path, keys[0]))) or \
        (not os.path.isdir(os.path.join(train_path, keys[1]))) or \
        (not os.path.isdir(os.path.join(train_path, keys[2]))) or \
        (not os.path.isdir(os.path.join(train_path, keys[3]))):
        raise FileNotFoundError(f"{train_path} is not a valid train directory initialized by Dataset_Gen.Dataset_Creation")
    d1 = datasets.load_from_disk(os.path.join(train_path, keys[0])).shuffle()
    d2 = datasets.load_from_disk(os.path.join(train_path, keys[1])).shuffle()
    d3 = datasets.load_from_disk(os.path.join(train_path, keys[2])).shuffle()
    d4 = datasets.load_from_disk(os.path.join(train_path, keys[3])).shuffle()
    aggr = interleave_datasets([d1, d2, d3, d4], probabilities=ratios, \
                               stopping_strategy="all_exhausted" if full else "first_exhausted")
    return aggr

def Concatenate_Datasets(path, keys = ["w+", "w-", "a+", "a-"]) -> Dataset:
    #a simple concatenation of datasets across four (?) sub-datasets within respective test/CV splits
    if (not os.path.isdir(os.path.join(path, keys[0]))) or \
        (not os.path.isdir(os.path.join(path, keys[1]))) or \
        (not os.path.isdir(os.path.join(path, keys[2]))) or \
        (not os.path.isdir(os.path.join(path, keys[3]))):
        raise FileNotFoundError(f"{path} is not a valid dataset cluster directory initialized by Dataset_Gen.Dataset_Creation")
    d1 = datasets.load_from_disk(os.path.join(path, keys[0]))
    d2 = datasets.load_from_disk(os.path.join(path, keys[1]))
    d3 = datasets.load_from_disk(os.path.join(path, keys[2]))
    d4 = datasets.load_from_disk(os.path.join(path, keys[3]))
    aggr = concatenate_datasets([d1, d2, d3, d4])
    return aggr
    