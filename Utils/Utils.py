import os
import random
import pickle

####
MAGICNUM = b"\xc4\x93Sp"
CHECKSUMCOEFF = (4,9,3,1,3,5,7,9)
CSCLEN = len(CHECKSUMCOEFF)
WEBIS_LEN = 7859 #1<-...> .. 7859<-...>
ALECS_OG_LEN = 157379 #157379 "text files"
####

def Save_Object(obj: ..., path: os.path):
    #safe-save object into file given by path. Overwrites.
    #makes directories up to path if not present.
    if len(os.path.dirname(path)) > 0 and not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    if os.path.isfile(path):
        f = open(path, "wb")
    else:
        f = open(path, "xb")
    
    f.write(MAGICNUM)
    pickled = pickle.dumps(obj)
    csum = 0
    for i, c in enumerate(pickled):
        csum += c * CHECKSUMCOEFF[i % CSCLEN]
        csum = csum if csum < 4294967296 else csum - 4294967296
    csum = hex(csum)[2:]
    while len(csum) < 8:
        csum = "0" + csum
    csum = bytes.fromhex(csum)
    f.write(csum)
    f.write(pickled)
    f.close()

def Recover_Object(path: os.path) -> ...:
    #read file and recover object. Only works with files saved using save_object(...)
    if not os.path.isfile(path):
        raise Exception(f"{path} is not a file")
    
    f = open(path, "rb")
    if f.read(len(MAGICNUM)) != MAGICNUM:
        raise Exception(f"{path} is not a file created by Save_Object")
    
    file_csum = int(f.read(4).hex(), base=16)
    content = f.read()
    f.close()
    calc_csum = 0
    for i,c in enumerate(content):
        calc_csum += c * CHECKSUMCOEFF[i % CSCLEN]
        calc_csum = calc_csum if calc_csum < 4294967296 else calc_csum - 4294967296
    
    if calc_csum != file_csum:
        raise Exception(f"Cannot read {path}; the file likely corrupted.")
    
    return pickle.loads(content)


def Webis_collect_matches(dsdir, page=0, page_size = 1000) -> list[(str, str, os.path, os.path)]:
    #only collects the "paraphrases marked as a paraphrase"
    #page: [0,7]; look at the next 1000 entries starting from page*page_size
    #page_size is not necessarily the length of the returned list, just the number of files looked
    
    #returns a list of (<original>, <paraphrase>, <orig_path>, <para_path>)
    r = []
    i = page * page_size
    while i < (page+1)*page_size and os.path.isfile(os.path.join(dsdir, str(i+1) + "-metadata.txt")):
        mn = os.path.join(dsdir, str(i+1))
        with open(mn + "-metadata.txt", "r") as f:
            v = f.read().split("\n")[3]
        if "Yes" in v:
            with open(mn + "-original.txt", "r", encoding="utf-8") as f:
                o = f.read()
            
            with open(mn + "-paraphrase.txt", "r", encoding="utf-8") as f:
                p = f.read()
            r.append((o,p, mn + "-original.txt", mn + "-paraphrase.txt"))
        i += 1
    return r

def ALECS_collect_matches(dir_og, dir_mg, page=0, page_size=1000) -> list[(str, str, os.path, os.path)]:
    # directories to og and mg to be provided
    # pagination done w.r.t. dir_og (not necessarily the length of the returned list)
    # if there are multiple machine-generated matches corresponding to one original,
    # they are both appended to the list.
    
    #returns a list of (<original>, <paraphrase>, <orig_path>, <para_path>)
    r = []
    ons = os.listdir(dir_og)
    i = page * page_size
    j = 0
    while j < page_size and i+j < ALECS_OG_LEN:
        c = i + j
        j += 1
        on = os.path.join(dir_og, ons[c])
        cand = ons[c][:ons[c].find(".txt")]
        cand = [cand + "0" + str(x) + ".txt" for x in range(4)]
        for fn in cand:
            if os.path.isfile(os.path.join(dir_mg, fn)):
                # match found; store the match
                pn = os.path.join(dir_mg, fn)
                with open(on, "r", encoding="utf-8") as f:
                    o = f.read()
                with open(pn, "r", encoding="utf-8") as f:
                    p = f.read()
                r.append((o,p, on, pn))
    return r

def ALECS_pick_non_match(dir_og, dir_mg, num_pairs) -> list[(str, str, os.path, os.path)]:
    # randomly grabs non-match pairs from ALECS dataset, with directory names provided.
    # generates num_pairs number of non-match pairs
    
    #returns a list of (<original>, <non-paraphrase>, <orig_path>, <nonp_path>)
    r = []
    ons = os.listdir(dir_og)
    mns = os.listdir(dir_mg)
    while len(r) < num_pairs:
        on = ons[random.randint(0,len(ons) - 1)]
        mn = mns[random.randint(0,len(mns) - 1)]
        cand = on[:on.find(".txt")]
        cand = [cand + "0" + str(x) + ".txt" for x in range(4)]
        if not (mn in cand):
            with open(os.path.join(dir_og, on), "r", encoding="utf-8") as f:
                o = f.read()
            with open(os.path.join(dir_mg, mn), "r", encoding="utf-8") as f:
                m = f.read()
            r.append((o,m, os.path.join(dir_og, on), os.path.join(dir_mg, mn)))
            
    return r