from torchvision import transforms

class BeamNode(object):
    def __init__(self, word_ids, scores, seq):
        '''
        :param wordIds:  Tensor, shape=(batch_size, 1), dtype=long
        :param scores:   Tensor, shape=(batch_size, 1), dtype=float
        :param seq:      List,   shape=(batch_size, length)
        '''
        self.word_ids = word_ids
        self.scores = scores
        self.seq = seq
        self.imgs = None
        self.targets = None
    
    def add_imgs(self, imgs):
        self.imgs = imgs
    
    def add_targets(self, targets):
        self.targets = targets