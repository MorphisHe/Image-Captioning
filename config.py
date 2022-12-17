def get_configs():
    train_params = {
        "device": "cuda",
        "max_epoch": 100,
        "output_dir": "outputs"
    }

    dataset_params = {
        "train_bs": 32,
        "test_bs": 100,
        "first_caption_only": True,
        "image_dir": "data/flickr8k/images/",
        "label_path": "data/flickr8k/results.csv",
        "delimiter": ","
    }

    encoder_params = {
        "embed_size": 256,
        "freeze_cnn": True
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
            "weight_decay": 0,
            "amsgrad": True
        }
    }

    return train_params, dataset_params, callback_params, optimizer_params, encoder_params, decoder_params
