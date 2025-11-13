import nltk

def to_wordlist(orig:str) -> list[str]:
    # removes all characters that are not lower/uppercase alphabet + numerics
    # converts orig into a list of lowercase words
    # only space characters break the words
    rl = []
    w = ""
    for c in orig:
        if c.isalnum():
            w += c.lower()
        elif c.isspace() and len(w) > 0:
            rl.append(w)
            w = ""
    if len(w) > 0:
        rl.append(w)
    return rl


class BLEU:
    # a naive BLEU removing all punctuations
    # by default uses NIST smoothing function. Otherwise provide a callable
    # smoothing function as a parameter during initialization
    def __init__(self, n=4, smoothing = None):
        self.n=4
        if smoothing == None:
            chencherry = nltk.translate.bleu_score.SmoothingFunction()
            self.sm = chencherry.method3
        else:
            self.sm = smoothing
    
    def score(self, cands, refs, info = False) -> list[float]:
        # expects cands a list of strings,
        # and refs also a list of strings
        # info does nothing.
        
        cands = [to_wordlist(x) for x in cands]
        refs = [[to_wordlist(x)] for x in refs]
        wts = [1/self.n for _ in range(self.n)]
        return [nltk.translate.bleu_score.sentence_bleu(ref, cand, weights=wts, 
             smoothing_function=self.sm) for (ref, cand) in zip(refs, cands)]



if __name__ == "__main__":
    X = BLEU()
    
    #source: 1000000 (og), 100000003 (mg), 2854103 (og) from ALECS
    f = "The Papiermark is the name given to the Germancurrency from 4 August 1914, when any link between the Goldmark and gold was abandoned. In particular, the name is used for certain banknotes issued during the period of hyperinflation in the Weimar Republic during 1922 and especially 1923. This set of Danzigmarks, in denominations of 100, 500 and 1000 kcal, was issued in 1922. These banknotes are partof the National Numismatic Collection at the Smithsonian Institution's National Museum of American History. During this period, the Papiermark was also issued by the Free City of Danzig. Then last of five series of the Danzig mark was another 1923 inflation issue, which consisted of denominations ranging 1 million to 10 billion issued from August to October 1923. The Danzig rul was replaced on 22 October 1923 by the Danzig gulden."
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."], 
                  ["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped plan bomb blasts. The cleric is believed to be a commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, and is linked to the Inter - Services Intelligence, the Pakistani spy agency."]))
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."],
                  [f]))