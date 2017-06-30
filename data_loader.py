import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision import transforms


data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}


class ImagesDataset(data.Dataset):
    def __init__(self, df, transform=None, loader=default_loader,
                 server=False):
        self.df = df
        self.transform = transform
        self.loader = loader
        self.server = server

    def __getitem__(self, index):
        row = self.df.iloc[index]

        target = row['class_'] - 1
        # target = target.astype(np.float32)
        path = row['path']
        if self.server:
            path = path.replace('/Users/illarionkhliestov/Datasets/',
                                '/home/illarion/datasets/')
        img = self.loader(path)
        img = np.array(img, np.float32)
        img = img / 255
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        n, _ = self.df.shape
        return n


class TestDatased(ImagesDataset):
    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['path']
        if self.server:
            path = path.replace('/Users/illarionkhliestov/Datasets/',
                                '/home/illarion/datasets/')

        img = self.loader(path)
        img = np.array(img, np.float32)
        img = img / 255
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def get_train_val_dataset(train_csv, val_csv, num_workers=1, batch_size=16, server=False):

    train_data_transform = data_transforms['train']
    val_data_transforms = data_transforms['val']

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_dataset = ImagesDataset(
        df=train_df,
        transform=train_data_transform,
        server=server)

    val_dataset = ImagesDataset(
        df=val_df,
        transform=val_data_transforms,
        server=server)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    return train_loader, val_loader


def get_test_dataset(test_csv, num_workers=1, batch_size=1, server=False):
    test_df = pd.read_csv(test_csv)
    test_dataset = TestDatased(
        df=test_df,
        transform=data_transforms['val'],
        server=server)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers)
    return test_loader
