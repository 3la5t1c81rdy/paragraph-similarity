import bert_score

class BERTScore:
    def __init__(self, model=None, idf=False):
        if model == None:
            self.model = "roberta-large"
        else:
            self.model = model
        self.idf = idf
    
    def score(self, cands, refs) -> list[float]:
        # expects cands a list of strings,
        # and refs also a list of strings
        
        # idf option is turned off
        res = bert_score.score(cands, refs, num_layers=17, model_type=self.model, idf=self.idf, lang="en")
        
        # return only the f-score components
        res = res[2].tolist()
        res = [float(x) for x in res] #and convert to float
        return res

if __name__ == "__main__":
    import time
    X = BERTScore()
    print(time.time())
    #source: 1000000 (og), 100000003 (mg), 2854103 (og) from ALECS
    f = "The Papiermark is the name given to the Germancurrency from 4 August 1914, when any link between the Goldmark and gold was abandoned. In particular, the name is used for certain banknotes issued during the period of hyperinflation in the Weimar Republic during 1922 and especially 1923. This set of Danzigmarks, in denominations of 100, 500 and 1000 kcal, was issued in 1922. These banknotes are partof the National Numismatic Collection at the Smithsonian Institution's National Museum of American History. During this period, the Papiermark was also issued by the Free City of Danzig. Then last of five series of the Danzig mark was another 1923 inflation issue, which consisted of denominations ranging 1 million to 10 billion issued from August to October 1923. The Danzig rul was replaced on 22 October 1923 by the Danzig gulden."
    
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."], 
                  ["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped plan bomb blasts. The cleric is believed to be a commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, and is linked to the Inter - Services Intelligence, the Pakistani spy agency."]))
    
    print(time.time())
    print(X.score(["A series of blasts occurred across the Hindu holy city of Varanasi on 7 March 2006. The cleric is believed to be deputy commander of a banned Bangladeshi Islamic militant group, Harkatul Jihad - al Islami, which is linked to the Inter - Services Intelligence, the Pakistani spy service. Fifteen people are reported to have been killed and as many as 101 others were injured. On 5 April 2006 the Indian police arrested six Islamic militants, including a cleric who helped organise multiple blasts."],
                  [f]))
    print(time.time())