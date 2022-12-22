import torch.nn as nn
import torchvision.models as models
from transformers import ViTImageProcessor, ViTModel


class EncoderCNN(nn.Module):
    # code from: https://github.com/tatwan/image-captioning-pytorch/blob/main/model.py with modification
    def __init__(self, embed_size, freeze_cnn=False):
        super(EncoderCNN, self).__init__()
        
        # pretrained model resnet50
        resnet = models.resnet152(pretrained=True) # ResNet: RGB order with pixels in [0, 1]
        
        # freeze cnn layer or not
        if freeze_cnn:
            for param in resnet.parameters():
                param.requires_grad_(False)
        
        # create encoding pipeline
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules) # delete the last fc layer
        
        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        '''
        `images`: (batch_size, #channel, height, width)
        '''
        features = self.resnet(images) # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1) # (batch_size, 2048)
        features = self.bn(self.embed(features)) # (batch_size, embed_size)

        return features


class EncoderViT(nn.Module):
    def __init__(self, embed_size, freeze_cnn=False):
        super(EncoderViT, self).__init__()

        # pretrained ViT model
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # freeze cnn layer or not
        if freeze_cnn:
            for param in self.vit.parameters():
                param.requires_grad_(False)

        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(768, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        '''
        `images`: (batch_size, #channel, height, width)
        '''
        features = self.vit(images).last_hidden_state # (batch_size, 197, 768)
        features = features[:,0,:].squeeze(1) # (batch_size, 768) 2nd index (0) is used to index CLS token embedding
        features = self.bn(self.embed(features)) # (batch_size, embed_size)

        return features
