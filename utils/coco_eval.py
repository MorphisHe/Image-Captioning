from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.spice.spice import Spice


class Scorer():
    def __init__(self):
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            #(Spice(), "SPICE")
        ]
    
    def compute_scores(self, refs, preds):
        '''
        refs: labels [[word1, word2, ...], [...], ...]
        preds: predictions in the form of [[word1, ...], [...], ...]
        '''
        assert len(refs) == len(preds)

        # process input
        new_refs = {}
        new_preds = {}
        for i, (ref, pred) in enumerate(list(zip(refs, preds))):
            new_refs[str(i)] = [' '.join(ref_i) for ref_i in ref]
            new_preds[str(i)] = [' '.join(pred)]

        # metrics
        total_scores = {}
        for scorer, method in self.scorers:
            score, _ = scorer.compute_score(new_refs, new_preds)
            if type(method) == list:
                total_scores["Bleu"] = score
            else:
                total_scores[method] = score
        
        bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4 = total_scores["Bleu"]
        ROUGE_L = total_scores["ROUGE_L"]
        METEOR = total_scores["METEOR"]
        CIDEr = total_scores["CIDEr"]
        #SPICE = total_scores["SPICE"]
        SPICE = 0

        return bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4, ROUGE_L, METEOR, CIDEr, SPICE


if __name__ == '__main__':
    ref = {
        '1':['go down the stairs and stop at the bottom .'],
        '2':['this is a cat.']
    }
    gt = {
        '1':['Walk down the steps and stop at the bottom. ', 'Go down the stairs and wait at the bottom.','Once at the top of the stairway, walk down the spiral staircase all the way to the bottom floor. Once you have left the stairs you are in a foyer and that indicates you are at your destination.'],
        '2':['It is a cat.','There is a cat over there.','cat over there.']
    }