import os
from torch.utils.data import random_split


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




if __name__ == "__main__":
    train, dev, test = get_data_split("data/flickr30k-images")
    print(len(train), len(dev), len(test))