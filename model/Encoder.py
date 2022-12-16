import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    # code from: https://github.com/tatwan/image-captioning-pytorch/blob/main/model.py with modification
    def __init__(self, embed_size, freeze=False):
        super(EncoderCNN, self).__init__()
        
        # pretrained model resnet50
        resnet = models.resnet50(pretrained=True) # ResNet: RGB order with pixels in [0, 1]
        
        # freeze cnn layer or not
        if freeze:
            for param in resnet.parameters():
                param.requires_grad_(False)
        
        # create encoding pipeline
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        '''
        `images`: (batch_size, #channel, height, width)
        '''
        features = self.resnet(images) # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1) # (batch_size, 2048)
        features = self.embed(features) # (batch_size, embed_size)

        return features


class EncoderViT(nn.Module):
    # TO BE IMPLEMENTED
    pass