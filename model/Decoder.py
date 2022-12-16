import torch
from torch import nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # code from: https://github.com/tatwan/image-captioning-pytorch/blob/main/model.py with modificati
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        #self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        
    
    def forward(self, features, captions):
        
        caption_embed = self.word_embedding(captions[:, :-1])
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed),1)
        output, self.hidden = self.lstm(caption_embed)
        #print(output.shape)
        output = self.fc(output)
        #print(self.embed_size, self.vocab_size)
        return output