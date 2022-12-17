import os
import torch
import time
from tqdm import tqdm
import numpy as np

from utils.logger import Logger
from utils.callbacks import EarlyStopping
from dataset.enums import END_SEQ, START_SEQ

import math
import nltk
import torch


class Trainer:
    def __init__(
        self,
        model, 
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        train_params,
        callback_params,
        optimizer_params,
    ):
        self.output_dir = train_params["output_dir"]
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.max_epochs = train_params["max_epoch"]
        self.device = train_params["device"]

        # init logger
        self.logger = Logger(self.output_dir)

        # init model
        self.model = model
        # print layer summary
        prev_layer_name = ""
        total_params = 0
        for name, param in self.model.named_parameters():
            layer_name = name.split(".")[0]
            if layer_name != prev_layer_name:
                prev_layer_name = layer_name
                self.logger.log_block("{:<70} {:<30} {:<30} {:<30}".format('Name','Weight Shape','Total Parameters', 'Trainable'))
            self.logger.log_message("{:<70} {:<30} {:<30} {:<30}".format(name, str(param.data.shape), param.data.numel(), param.requires_grad))
            total_params += np.prod(param.data.shape)
        self.logger.log_block(f"Total Number of Paramters: {total_params}")
        self.logger.log_line()

        # log dataloaders
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.test_dataloader = test_dataloader
        self.total_train_batch = len(self.train_dataloader)
        self.total_test_batch = len(self.test_dataloader)
        self.total_dev_batch = len(self.dev_dataloader)
        self.ten_percent_train_batch = max(self.total_train_batch // 10, 1) # use to log step loss
        self.logger.log_block(f"Training Dataset Size: {len(self.train_dataloader.dataset)}")
        self.logger.log_message(f"Training Dataset Total Batch#: {self.total_train_batch}")
        self.logger.log_block(f"Dev Dataset Size: {len(self.dev_dataloader.dataset)}")
        self.logger.log_message(f"Dev Dataset Total Batch#: {self.total_dev_batch}")
        self.logger.log_message(f"Test Dataset Size: {len(self.test_dataloader.dataset)}")
        self.logger.log_message(f"Test Dataset Total Batch#: {self.total_test_batch}")

        # init callback [early stopping]
        self.callbacks = EarlyStopping(self.logger, self.output_dir, **callback_params)

        # init optimizer and loss
        param_dict = [{
            "params": self.model.parameters(), 
            "lr": optimizer_params["lr"], 
        }]
        self.optimizer = getattr(torch.optim, optimizer_params["type"])(param_dict, **optimizer_params["kwargs"])
        self.criterion = torch.nn.CrossEntropyLoss()
    

        # put model to device
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # log all configs
        self._log_configs(train_params, optimizer_params)
        
    def _log_configs(
        self,
        train_params,
        optimizer_params
    ):
        # log trainer kwargs
        self.logger.log_line()
        self.logger.log_message("Trainer Kwargs:")
        self.logger.log_new_line()
        for k, v in train_params.items():
            self.logger.log_message("{:<30} {}".format(k, v))

        # log optimizer kwargs
        self.logger.log_line()
        self.logger.log_message(f"Optimizer: {optimizer_params}")

        # log Callbacks kwargs
        self.logger.log_line()
        self.logger.log_message(f"Callbacks: {self.callbacks.__class__.__name__}")
        self.logger.log_new_line()
        self.logger.log_message("{:<30} {}".format('save_final_model', self.callbacks.save_final_model))
        self.logger.log_message("{:<30} {}".format('patience', self.callbacks.patience))
        self.logger.log_message("{:<30} {}".format('threshold', self.callbacks.threshold))
        self.logger.log_message("{:<30} {}".format('mode', self.callbacks.mode))

    def train(self):
        for epoch in range(self.max_epochs):
            # train one epoch
            self.cur_epoch = epoch
            self.logger.log_line()
            self.train_one_epoch()
            
            # eval one epoch
            self.logger.log_line()
            self.eval_one_epoch()

        self.logger.log_block(f"Max epoch reached. Best f1-score: {self.callbacks.best_score:.4f}")
        self.eval_best_model_on_testdataset()
        exit(1)

    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        ten_percent_batch_loss = 0 # use to accumlate training loss every 10% of an epoch
        start_time = time.time()
        
        for batch_idx, (images, captions, lengths) in enumerate(self.train_dataloader):
            images, captions, lengths = images.to(self.device), captions.to(self.device), lengths
            loss = self.train_one_step(images, captions, lengths)
            epoch_loss += loss
            ten_percent_batch_loss += loss

            if (batch_idx+1)%self.ten_percent_train_batch == 0:
                ten_percent_avg_loss = ten_percent_batch_loss/self.ten_percent_train_batch
                message = f"Epoch {self.cur_epoch} - iter {batch_idx}/{self.total_train_batch} - 10% Avg loss {ten_percent_avg_loss:.5f}"
                self.logger.log_message(message=message)
                ten_percent_batch_loss = 0
        
        end_time = time.time()
        avg_loss = epoch_loss/self.total_train_batch
        epoch_time = (end_time-start_time)/60
        self.logger.log_message(f"Epoch #{self.cur_epoch}: Epoch Average Loss {avg_loss:.5f} - Epoch Training Time: {epoch_time:.2} min(s)")

    def train_one_step(self, images, captions, lengths):
        self.optimizer.zero_grad()
        outputs = self.model(images, captions, lengths) # (batch_size, padded_seq_length, vocab_size)

        # remove padding in loss calculation
        unpadded_outputs = []
        unpadded_captions = []
        for length, output, caption in zip(lengths, outputs, captions):
            unpadded_outputs.append(output[:length,:]) # (length, vocab_size)
            unpadded_captions.append(caption[:length].unsqueeze(1)) # (length, 1)

        unpadded_outputs = torch.vstack(unpadded_outputs) # (total_num_words, vocab_size)
        unpadded_captions = torch.vstack(unpadded_captions) # (total_num_words, 1)

        loss = self.criterion(
            unpadded_outputs,
            unpadded_captions.squeeze()
        )
        
        # steps
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_one_epoch(self):
        self.model.eval()
        test_loss = 0
        start_time = time.time()
        
        # bleu eval using greedy search
        vocab = self.train_dataloader.dataset.vocab
        end_seq_val = vocab.word2idx[END_SEQ]
        with torch.no_grad():
            captions_decoded = []
            prediction_decoded = []
            for images, captions, lengths in tqdm(self.dev_dataloader, desc="Eval Devset"):
                # decode ground truth captions
                for caption in captions:
                    encoded_caption = caption.cpu().numpy().astype(int).tolist()
                    end_seq_idx = encoded_caption.index(end_seq_val)
                    encoded_caption = encoded_caption[:end_seq_idx+1]
                    decoded_caption = vocab.decode_seq(encoded_caption)
                    captions_decoded.append([decoded_caption])

                # get prediction   
                images, captions, lengths = images.to(self.device), captions.to(self.device), lengths
                img_feats = self.model.encoder(images)
                outputs = self.model.decoder(img_feats, captions, lengths)

                # remove padding in loss calculation
                unpadded_outputs = []
                unpadded_captions = []
                for length, output, caption in zip(lengths, outputs, captions):
                    unpadded_outputs.append(output[:length,:]) # (length, vocab_size)
                    unpadded_captions.append(caption[:length].unsqueeze(1)) # (length, 1)

                unpadded_outputs = torch.vstack(unpadded_outputs) # (total_num_words, vocab_size)
                unpadded_captions = torch.vstack(unpadded_captions) # (total_num_words, 1)

                loss = self.criterion(
                    unpadded_outputs,
                    unpadded_captions.squeeze()
                )
                test_loss += loss.item()

                # greedy search
                max_seq_length = max(lengths)
                generated_seqs = self.model.decoder.generate_sequence(img_feats, max_seq_length=max_seq_length)
                generated_seqs = generated_seqs.cpu().numpy().astype(int).tolist()
                for generated_seq in generated_seqs:
                    end_seq_idx = -1 # keep whole sequence if END_SEQ not generated by model
                    if end_seq_val in generated_seq:
                        # check if END_SEQ is generated by the model
                        end_seq_idx = generated_seq.index(end_seq_val)
                    generated_seq = generated_seq[:end_seq_idx+1]
                    if not len(generated_seq):
                        # if generated empty sequence, append a end_seq to it
                        generated_seq = [end_seq_val]
                    decoded_gen_seq = vocab.decode_seq(generated_seq)
                    prediction_decoded.append(decoded_gen_seq)
        
        # compute bleu score
        bleu_score = nltk.bleu_score.corpus_bleu(captions_decoded, prediction_decoded, weights=(0.33, 0.33, 0.33, 0.0))

        end_time = time.time()
        avg_loss = test_loss/self.total_test_batch
        epoch_time = (end_time-start_time)/60
        self.logger.log_message(f"Eval Devset: Epoch #{self.cur_epoch}: Average Loss {avg_loss:.5f} - BLEU Score: {bleu_score:.5f} - Epoch Testing Time: {epoch_time:.2} min(s)")

        # saving best model and early stopping
        if not self.callbacks(self.model, bleu_score):
            self.eval_best_model_on_testdataset()
            exit(1)

    def eval_best_model_on_testdataset(self):
        self.model.load_state_dict(torch.load(os.path.join(self.output_dir, "best-model.pt")))
        self.model.eval()
        test_loss = 0
        start_time = time.time()
        
        # bleu eval using greedy search
        vocab = self.test_dataloader.dataset.vocab
        end_seq_val = vocab.word2idx[END_SEQ]
        with torch.no_grad():
            captions_decoded = []
            prediction_decoded = []
            for images, captions, lengths in tqdm(self.test_dataloader, desc="Eval Testset"):
                # decode ground truth captions
                for caption in captions:
                    encoded_caption = caption.cpu().numpy().astype(int).tolist()
                    end_seq_idx = encoded_caption.index(end_seq_val)
                    encoded_caption = encoded_caption[:end_seq_idx+1]
                    decoded_caption = vocab.decode_seq(encoded_caption)
                    captions_decoded.append([decoded_caption])

                # get prediction   
                images, captions, lengths = images.to(self.device), captions.to(self.device), lengths
                img_feats = self.model.encoder(images)
                outputs = self.model.decoder(img_feats, captions, lengths)

                # remove padding in loss calculation
                unpadded_outputs = []
                unpadded_captions = []
                for length, output, caption in zip(lengths, outputs, captions):
                    unpadded_outputs.append(output[:length,:]) # (length, vocab_size)
                    unpadded_captions.append(caption[:length].unsqueeze(1)) # (length, 1)

                unpadded_outputs = torch.vstack(unpadded_outputs) # (total_num_words, vocab_size)
                unpadded_captions = torch.vstack(unpadded_captions) # (total_num_words, 1)

                loss = self.criterion(
                    unpadded_outputs,
                    unpadded_captions.squeeze()
                )
                test_loss += loss.item()

                # greedy search
                max_seq_length = max(lengths)
                generated_seqs = self.model.decoder.generate_sequence(img_feats, max_seq_length=max_seq_length)
                generated_seqs = generated_seqs.cpu().numpy().astype(int).tolist()
                for generated_seq in generated_seqs:
                    end_seq_idx = -1 # keep whole sequence if END_SEQ not generated by model
                    if end_seq_val in generated_seq:
                        # check if END_SEQ is generated by the model
                        end_seq_idx = generated_seq.index(end_seq_val)
                    generated_seq = generated_seq[:end_seq_idx+1]
                    if not len(generated_seq):
                        # if generated empty sequence, append a end_seq to it
                        generated_seq = [end_seq_val]
                    decoded_gen_seq = vocab.decode_seq(generated_seq)
                    prediction_decoded.append(decoded_gen_seq)
        
        # compute bleu score
        bleu_score = nltk.bleu_score.corpus_bleu(captions_decoded, prediction_decoded, weights=(0.33, 0.33, 0.33, 0.0))

        end_time = time.time()
        avg_loss = test_loss/self.total_test_batch
        epoch_time = (end_time-start_time)/60
        self.logger.log_message(f"Test Devset: Epoch #{self.cur_epoch}: Average Loss {avg_loss:.5f} - BLEU Score: {bleu_score:.5f} - Epoch Testing Time: {epoch_time:.2} min(s)")