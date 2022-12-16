def get_configs():
    train_params = {
        "device": "cuda",
        "max_epoch": 100,
        "output_dir": "outputs"
    }

    dataset_params = {
        "train_bs": 16,
        "test_bs": 256,
        "first_caption_only": False,
        "image_dir": "data/flickr30k-images/",
        "label_path": "results.csv"
    }

    encoder_params = {
        "embed_size": 256,
        "freeze_cnn": False
    }

    decoder_params = {
        "embed_size": encoder_params["embed_size"],
        "hidden_size": 512,
        "num_layers": 1
    }

    callback_params = {
        "patience": 5,
        "save_final_model": True,
    }

    optimizer_params = {
        "lr": 0.01,
        "type": "Adam",
        "kwargs": {
            "weight_decay": 0.1,
            "amsgrad": True
        }
    }

    return train_params, dataset_params, callback_params, optimizer_params, encoder_params, decoder_params
