import sys
sys.path.append("../")

import os
import pandas as pd
from torch.utils.data import random_split
from dataset.flickr30k import Vocabulary


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

    return train, dev, test


def get_vocabulary(label_path):
    vocab = Vocabulary()

    # get labels that belong to this dataset
    df = pd.read_csv(label_path, delimiter="|")
    records = df.to_dict("record")
    for record in records:
        # lowercase, remove special chars
        print(record["image_name"])
        print(record[" comment"])
        caption = record[" comment"].strip()
        tokens = caption.split()
        tokens = [token.lower() for token in tokens if token.isalnum()]
        
        # add to vocab
        for token in tokens:
            vocab.add_word(token)
    
    return vocab


#def get_dataloader()



if __name__ == "__main__":
    train, dev, test = get_data_split("../data/flickr30k-images")
    print(len(train), len(dev), len(test))

    vocab = get_vocabulary("../data/results.csv")
    print(len(vocab))