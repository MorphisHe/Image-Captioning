import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from texar.torch.utils import beam_search



class DecoderRNN(nn.Module):
    # code from: https://github.com/tatwan/image-captioning-pytorch/blob/main/model.py with modification
    def __init__(self, use_gru, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # create layers
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        if use_gru:
            self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:    
            self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)
    
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
        packed_cap_embed = pack_padded_sequence(
                                caption_embed, lengths, 
                                batch_first=True,
                                enforce_sorted=False) # data: (total_words[padded], embed_size)
        hiddens, _ = self.lstm(packed_cap_embed) # data: (total_words[padded], hidden_size)
        #hiddens, _ = pad_packed_sequence(hiddens, batch_first=True) # unpack: (batch_size, padded_seq_length, hidden_size)
        #outputs = self.linear(hiddens) # (batch_size, padded_seq_length, vocab_size)
        outputs = self.linear(hiddens[0]) # (total_words[padded], vocab_size)

        return outputs

    def generate_sequence(self, features, states=None, max_seq_length=20):
        '''
        Gready Search: for inference time
        `features`: (batch_size, embed_size) encoded image feature from Encoder model
        '''
        generated_seqs = []
        inputs = features.unsqueeze(1) # (batch_size, 1, embed_size)
        for i in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states) # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1)) # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1) # predicted: (batch_size)
            generated_seqs.append(predicted)
            inputs = self.word_embedding(predicted) # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1) # inputs: (batch_size, 1, embed_size)
        generated_seqs = torch.stack(generated_seqs, 1) # generated_seqs: (batch_size, max_seq_length)
        
        return generated_seqs

    def _symbols_to_logits_fn(self, inputs, states):
        '''
        function that can take the currently decoded symbols and return the logits for the next symbol

        `inputs` (batch_size, decoded_ids[size of current timestamp])
        `outputs` (batch_size, vocab_size)
        '''
        inputs = inputs[:,-1].squeeze() # (beam_size)
        states = (states[0].squeeze().unsqueeze(0), states[1].squeeze().unsqueeze(0))
        inputs = self.word_embedding(inputs) # inputs: (batch_size, embed_size)
        inputs = inputs.unsqueeze(1) # inputs: (batch_size, 1, embed_size)
        hiddens, states = self.lstm(inputs, states) # hiddens: (batch_size, 1, hidden_size)
        states = (states[0].squeeze(), states[1].squeeze())
        outputs = self.linear(hiddens.squeeze(1)) # outputs:  (batch_size, vocab_size)

        return outputs, states

    def beam_search(self, features, max_seq_length=20, beam_size=3, alpha=0):
        END_SEQ = 3

        # get initial node symbols [ideally all START_SEQ] using encoder feature
        inputs = features.unsqueeze(1) # (batch_size, 1, embed_size)
        hiddens, states = self.lstm(inputs) # hiddens: (batch_size, 1, hidden_size)
        outputs = self.linear(hiddens.squeeze(1)) # outputs:  (batch_size, vocab_size)
        _, predicted = outputs.max(1) # predicted: (batch_size)

        # beam search
        final_ids, final_probs = beam_search.beam_search(
            symbols_to_logits_fn=self._symbols_to_logits_fn,
            initial_ids=predicted,
            beam_size=beam_size,
            decode_length=max_seq_length+10,
            vocab_size=self.vocab_size,
            alpha=alpha,
            states=states,
            eos_id=END_SEQ
        )# (batch_size, beam_size, max_seq_length), (batch_size, beam_size)
        
        _, best_seq_idx = final_probs.max(1)
        best_seq = final_ids[:, best_seq_idx ,:].squeeze(1) # (batch_size, max_seq_length)
        best_seq = torch.stack([best_seq, torch.tensor([[END_SEQ]], device=inputs.device)], dim=1)

        return best_seq