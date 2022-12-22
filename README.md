# DL-ImageCaptioning

### Set up enviornment
```bash
pip3 install -r requirements.txt
git clone https://github.com/asyml/texar-pytorch.git 
cd texar-pytorch
!pip install -e .
cd ..
```

<br>


### Model Configs: modify config.py
```py
encoder_params = {
    "use_vit": False, # set to True to use ViT instead of ResNet-152
    "embed_size": 512,
    "freeze_cnn": True
}

decoder_params = {
    "use_gru": False, # set to True to use GRU instead of LSTM
    "embed_size": encoder_params["embed_size"],
    "hidden_size": 512,
    "num_layers": 1,
    "use_beam_search": False, # set to True to use Beam Search instead of Greedy
    "beam_size": 5 # set to beam size when using Beam Search
}
```

<br>

### Training:
- modify `image_dir` and `label_path` to match your data path
```bash
python3 train.py # train model
```

<br>

### Plot Loss:
- we have already prepared model training log file in `training_logs` folder to demo loss plotting
```bash
python3 plot_loss.py
```

<br>

### Training Log:
`training_logs` stores all model training logs. Since model checkpoint is too large, we can't push them to github repo. However training log shows loss and set of metric for training the model. Each log file is named with the architecture used for that particular training. Ex: VIT-LSTM-BEAM3 uses ViT encoder and LSTM decoder with Beam Search and Beam Size of 3.