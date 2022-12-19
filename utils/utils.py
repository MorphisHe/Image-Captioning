import sys
sys.path.append("../")

import os
import re
import pandas as pd
from torch.utils.data import random_split, DataLoader
from dataset.flickr_dataset import Vocabulary, FlickrConvRNN, BatchCollateFn

import nltk
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score


def get_data_split(image_dir, trainset_ratio=0.8):
    # get image ids
    image_ids = os.listdir(image_dir)
    image_ids = [image_id for image_id in image_ids if ".jpg" in image_id]

    # compute train, dev, test size
    train_size = int(trainset_ratio * len(image_ids))
    train_dev_size = len(image_ids) - train_size
    dev_size = int(0.5 * train_dev_size)
    test_size = train_dev_size - dev_size

    # get split
    train, dev, test = random_split(image_ids, [train_size, dev_size, test_size])

    return list(train), list(dev), list(test)


def get_vocabulary(label_path, delimiter):
    vocab = Vocabulary()

    # get labels that belong to this dataset
    df = pd.read_csv(label_path, delimiter=delimiter)
    records = df.to_dict("record")
    for record in records:
        # lowercase, remove special chars
        caption = record[" comment"].strip()
        tokens = caption.split()
        tokens = [re.sub(r'[^a-zA-Z0-9]', '', token.lower()) for token in tokens if token.isalnum()]
        tokens = [token for token in tokens if len(token)] # remove empty token
        
        # add to vocab
        for token in tokens:
            vocab.add_word(token)
    
    return vocab


def get_dataloader(train_ids, dev_ids, test_ids, 
                   image_dir, label_path, vocab, 
                   first_caption_only, train_bs, 
                   test_bs, delimiter,
                   transform_train, transform_test):
    # create dataset
    train_ds = FlickrConvRNN(
        image_dir=image_dir,
        image_ids=train_ids,
        label_path=label_path,
        transform=transform_train,
        vocab=vocab,
        first_caption_only=first_caption_only,
        delimiter= delimiter
    )
    dev_ds = FlickrConvRNN(
        image_dir=image_dir,
        image_ids=dev_ids,
        label_path=label_path,
        transform=transform_test,
        vocab=vocab,
        first_caption_only=first_caption_only,
        delimiter= delimiter
    )
    test_ds = FlickrConvRNN(
        image_dir=image_dir,
        image_ids=test_ids,
        label_path=label_path,
        transform=transform_test,
        vocab=vocab,
        first_caption_only=first_caption_only,
        delimiter= delimiter
    )
    
    # create batch collate function
    collate_fn = BatchCollateFn()

    # create dataloaders
    train_dataloader = DataLoader(train_ds, batch_size=train_bs, shuffle=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_ds, batch_size=test_bs, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_ds, batch_size=test_bs, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader


def eval_image_captioning(captions_decoded, prediction_decoded):
    """
    captions_decoded in the form of [[list values of str],[...],...]
    prediction_decoded in the form of [[list values of str],[...],...]
    """
    bleu_score_1 = nltk.bleu_score.corpus_bleu(captions_decoded, prediction_decoded, weights=(1.0, 0.0, 0.0, 0.0))
    bleu_score_2 = nltk.bleu_score.corpus_bleu(captions_decoded, prediction_decoded, weights=(0.5, 0.5, 0.0, 0.0))
    bleu_score_3 = nltk.bleu_score.corpus_bleu(captions_decoded, prediction_decoded, weights=(0.33, 0.33, 0.33, 0.0))
    bleu_score_4 = nltk.bleu_score.corpus_bleu(captions_decoded, prediction_decoded, weights=(0.25, 0.25, 0.25, 0.25))
    rouge = Rouge()

    rouge_scores = []
    meteor_scores = []
    for reference, candidate in zip(captions_decoded, prediction_decoded):
        score1 = rouge.get_scores(reference[0], candidate[0])[0]['rouge-l']['f']
        rouge_scores.append(score1)

        score2 = meteor_score(reference[0], candidate[0])
        meteor_scores.append(score2)

    ROUGE_L = sum(rouge_scores) / len(rouge_scores)
    METEOR = sum(meteor_scores) / len(meteor_scores)

    return bleu_score_1, bleu_score_2, bleu_score_3, bleu_score_4, ROUGE_L, METEOR




if __name__ == "__main__":
    train, dev, test = get_data_split("../data/flickr30k-images")
    print(len(train), len(dev), len(test))

    vocab = get_vocabulary("../data/results.csv")
    print(len(vocab))