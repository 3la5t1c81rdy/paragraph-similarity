import os
import random
import pickle
import numpy as np

####
MAGICNUM = b"\xc4\x93Sp"
CHECKSUMCOEFF = (4,9,3,1,3,5,7,9)
CSCLEN = len(CHECKSUMCOEFF)
WEBIS_LEN = 7859 #1<-...> .. 7859<-...> => 4067 positive samples
ALECS_OG_LEN = 157379 #157379 "text files" => 101895 positive samples
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

###horrendously slow but does the job
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


####
def MSE(A,B, weight = None):
    A = np.array(A)
    B = np.array(B)
    if weight is None:
        return np.square(A-B).mean()
    else:
        return (np.square(A-B) * weight).sum() / np.sum(weight)

def Pearson_Corr(*pred_res:list[float]) -> list[float]:
    #computes correlation if given args (pred_res) of identical lengths.
    #denoting first list given as (1), second as (2), ..., returns list correlations of:
    #    (1,2), (1,3), ..., (1,n), (2,3), ..., (2,n), ..., (n-1,n)
    if len(pred_res) < 2:
        return []
    l = len(pred_res[0])
    if l < 2:
        raise ValueError(f"length {l} is unsupported")
    for i in range(1, len(pred_res)):
        if len(pred_res[i]) != l:
            raise ValueError(f"{i}th list has length {len(pred_res[i])} which does not match the length of the first list ({l})")
    
    r = np.corrcoef(pred_res)
    rl = []
    for i in range(len(pred_res)):
        for j in range(i+1, len(pred_res)):
            rl.append(float(r[i,j]))
    
    return rl

class RangeMap:
    #a custom multicalibration mapping mask.
    #can MAP the output value of a function into a "calibrated" value
    def __init__(self, bins, v_range, mapping):
        #initializes RangeMap with [bins] number of "bins", across [v_range] values
        #v_range must be a tuple-like; (a,b,...) is taken as an inclusive range between a and b
        #    ... with "bins" number of subdivisions (-1 if you cound the singleton maximum bin).
        #    ... each subdivisions include [a,a+w), [a+w,a+2w), ..., [b-w, b), [b] for w the bin-width
        if not (isinstance(bins, int) and isinstance(v_range[0], float) and isinstance(v_range[1], float) and len(mapping) == bins + 1):
            raise ValueError(f"Invalid RangeMap parameters. ")
        self.minimum = v_range[0] #should be the minimum value of the function being calibrated
        self.maximum = v_range[1] #should be the maximum value of the function being calibrated
        if self.minimum >= self.maximum:
            raise ValueError("Invalid RangeMap parameters.")
        self.bins = bins
        self.mapping = mapping #in reality, should be a list of floats corresponding to the calibrated return value
        
    def map(self, f_x) -> float:
        #returns a calibrated value according to the value stored in mapping.
        if f_x < self.minimum:
            #outside of calibration range; clip into the minimum
            reg = 0
        elif f_x > self.maximum:
            #outside of calibration range; clip into the maximum
            reg = self.bins
        else:
            width = self.maximum - self.minimum
            
            #find bin
            reg = min(int((self.bins * (f_x - self.minimum)) / width), self.bins)
        
        if not (isinstance(self.mapping[reg], int) or isinstance(self.mapping[reg], float)):
            #no mapping registered
            return f_x
        return self.mapping[reg]
    
    def clip(self, clear = False):
        #internally "clips" the mapping to only output the value clipped to the lower bound
        #of the bin
        #if clear is set to true, overrides current value; otherwise, only cleans the "non-numerical" maps
        width = (self.maximum - self.minimum) / self.bins
        for i in range(self.bins):
            if clear or (not (isinstance(self.mapping[i], int) or isinstance(self.mapping[i], float))):
                self.mapping[i] = self.minimum + (i * width)
        self.mapping[self.bins] = self.maximum
    
    def output_vals(self) -> set[float]:
        #returns a list of all possible values, IF CLIPPED. Otherwise returns None
        rs = set()
        for x in self.mapping:
            if x is None:
                return None
            rs.add(x)
        
        return rs
    
    def patch_step(self, v_t, v_prime) -> bool:
        #patches all bins outputting (exactly) v_t into v_prime
        #returns whether patch occurred
        c = False
        for i in range(len(self.mapping)):
            if self.mapping[i] == v_t:
                self.mapping[i] = v_prime
                c = True
        
        return c

#probably optimizable, but can't bother lol
def Fit_Calibration(preds, labels, bins=100, v_range = (0.0,1.0), alpha = 1e-7) -> RangeMap:
    #approx. calibration via "binning" the initial function output into [bins] + 1 number of bins
    #    across v_range
    #each of preds and labels must be of equal length
    #preds is a list of predictions
    #labels is the true labels matched to the above predictions
    #
    #A simple implementation of the algorithm presented in class
    #
    #CAUTION: for test purposes, this calibration better only be applied to the Train set.
    #    => measure performance on CV and Test, promptly.
    if len(preds) != len(labels):
        raise ValueError(f"Unmatching lengths between preds ({len(preds)}) and labels ({len(labels)}).")
    
    rm = RangeMap(bins, v_range, [None for _ in range(bins + 1)]) #initial "identity" map
    
    rm.clip() #clip
    while True:
        #check if alpha-calibrated
        V = rm.output_vals()
        D = {v:[] for v in V}
        #populate D
        for i in range(len(preds)):
            v = rm.map(preds[i])
            D[v].append(labels[i])
        
        #find v_t, (argmax) while also checking calibration (K_2)
        v_t = None #the argmax
        maxval = None
        
        K = 0
        for v in V:
            if len(D[v]) == 0:
                continue #safely be continued; if every len(..) are 0, loop broken.
            cwt = (len(D[v])) * ((v - np.mean(D[v]))**2)
            
            K += cwt / len(preds)
            # K += P[f(x) = v] * (v - E[y | ..])^2
            
            if maxval is None or maxval < cwt:
                maxval = cwt
                v_t = v
        
        if K <= alpha:
            #is calibrated
            break
        
        rm.patch_step(v_t, np.mean(D[v_t])) #patch
    
    return rm
if __name__ == "__main__":
    print(Pearson_Corr([0,0.1,0.2,0.5], [1,0.8,0.4,0.3],[0.1,0.1,0.1,0]))
    #prints [-0.8873274787391783, -0.9258200997725514, 0.6557632539969139]
    rm = Fit_Calibration([0,0.1,0.2,0.9,0.0,1], [0,0.1,0.3,0.1,0.5,1], 10, (0.0,1.0))
    
    print(rm.mapping) #mapping([0,0.1,0.2,0.9,0.0,1]) is calibrated to the labels [0,0.1,0.3,0.1,0.5,1]