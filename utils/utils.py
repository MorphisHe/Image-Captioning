import torch
import deeplake

class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_flickr30k(trainset_ratio=0.8, train_bs=16, test_bs=128, transform_train=None, transform_test=None):
    trainset = deeplake.load('hub://activeloop/flickr30k')
    
    # compute train, dev, test size
    train_size = int(trainset_ratio * len(trainset))
    train_dev_size = len(trainset) - train_size
    dev_size = int(0.5 * train_dev_size)
    test_size = train_dev_size - dev_size

    # train, dev, test split
    trainset, devset, testset = torch.utils.data.random_split(trainset, [train_size, devset, test_size])

    # create subset so that data transformation don't share across each split
    trainset = DatasetFromSubset(trainset, transform=transform_train)
    devset = DatasetFromSubset(devset, transform=transform_test)
    testset = DatasetFromSubset(testset, transform=transform_test)

    # create data loader
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)
    devloader = torch.utils.data.DataLoader(devset, batch_size=test_bs, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    return trainloader, devloader, testloader

