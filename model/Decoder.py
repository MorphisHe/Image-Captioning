import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class DecoderRNN(nn.Module):
    # code from: https://github.com/tatwan/image-captioning-pytorch/blob/main/model.py with modification
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()

        # create layers
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        '''
        `features`: (batch_size, embed_size) encoded image feature from Encoder model
        `captions`: (batch_size, padded_seq_length) encoded captions
        `lengths`: (batch_size, ) lengths of each caption before padding
        '''
        # caption embedding
        caption_embed = self.word_embedding(captions) # (batch_size, padded_seq_length, embed_size)

        # ingestion step, appending image feature to caption embedding
        # (batch_size, padded_seq_length+1, embed_size) 
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed), 1)
        
        # rnn
        self.lstm.flatten_parameters()
        packed_cap_embed = pack_padded_sequence(
                                caption_embed, lengths, 
                                batch_first=True,
                                enforce_sorted=False) # data: (total_words[padded], embed_size)
        hiddens, _ = self.lstm(packed_cap_embed) # data: (total_words[padded], hidden_size)
        hiddens, _ = pad_packed_sequence(hiddens, batch_first=True) # unpack: (batch_size, padded_seq_length, hidden_size)
        outputs = self.linear(hiddens) # (batch_size, padded_seq_length, vocab_size)

        return outputs

    def generate_sequence(self, features, states=None, max_seq_length=20):
        '''
        Gready Search: for inference time
        `features`: (batch_size, embed_size) encoded image feature from Encoder model
        '''
        sampled_ids = []
        inputs = features.unsqueeze(1) # (batch_size, 1, embed_size)
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states) # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1)) # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1) # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.word_embedding(predicted) # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1) # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1) # sampled_ids: (batch_size, max_seq_length)
        
        return sampled_ids