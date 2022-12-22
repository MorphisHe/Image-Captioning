import torch
import torchvision.transforms as transforms

import numpy as np
import random
from config import get_configs
from trainer import Trainer
from utils.utils import get_data_split, get_vocabulary, get_dataloader
from model.image_captioning_model import ConvRNN


def get_train_test_trans():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize((0.458, 0.446, 0.404), (0.280, 0.273, 0.287)), #flickr8k
        # transforms.Normalize((0.444, 0.421, 0.385), (0.285, 0.277, 0.286)), #flickr30k

    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        #transforms.Normalize((0.458, 0.446, 0.404), (0.280, 0.273, 0.287)) #flickr8k
        #transforms.Normalize((0.444, 0.421, 0.385), (0.285, 0.277, 0.286)), #flickr30k
    ])

    return transform_train, transform_test



if __name__ == "__main__":
    seed = 6953
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # get configs
    train_params, dataset_params, callback_params, optimizer_params, encoder_params, decoder_params = get_configs()

    # get data augmentations
    transform_train, transform_test = get_train_test_trans()

    # get vocab
    vocab = get_vocabulary(dataset_params["label_path"], dataset_params["delimiter"])

    # get model
    use_vit = encoder_params.pop("use_vit")
    use_beam_search = decoder_params.pop("use_beam_search")
    beam_size = decoder_params.pop("beam_size")
    model = ConvRNN(use_vit, encoder_params, decoder_params, vocab_size=len(vocab))

    # get dataloaders
    if use_beam_search:
        # beam only support test batch_size of 1!!!
        dataset_params["test_bs"] = 1
    train_ids, dev_ids, test_ids = get_data_split(dataset_params["image_dir"])
    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(
        train_ids, dev_ids, test_ids, **dataset_params, vocab=vocab,
        use_beam_search=use_beam_search, beam_size=beam_size,
        transform_train=transform_train, transform_test=transform_test
    )

    # get trainer
    trainer = Trainer(model, train_dataloader, dev_dataloader, test_dataloader, 
                      train_params, callback_params, optimizer_params)
    trainer.train()

