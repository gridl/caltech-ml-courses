import os

import pandas as pd
import torch
import torch.utils.data as data
from torchvision.datasets.folder import default_loader
from torchvision import transforms


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop((64, 64), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}


class ImagesDataset(data.Dataset):
    def __init__(self, df, root_dir, transform=None, loader=default_loader):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        row = self.df.iloc[index]

        target = row['class_'] - 1
        # target = target.astype(np.float32)
        path = row['path']
        path = os.path.join(self.root_dir, path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        n, _ = self.df.shape
        return n


class ImagesTestDatased(ImagesDataset):
    def __getitem__(self, index):
        row = self.df.iloc[index]
        path = row['path']
        path = os.path.join(self.root_dir, path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, path


def get_train_val_test_dataset(train_csv, val_csv, test_csv, root_dir,
                               num_workers=1, batch_size=16):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = ImagesDataset(
        df=train_df,
        transform=data_transforms['train'],
        root_dir=root_dir)

    val_dataset = ImagesDataset(
        df=val_df,
        transform=data_transforms['val'],
        root_dir=root_dir)

    test_dataset = ImagesTestDatased(
        df=test_df,
        transform=data_transforms['val'],
        root_dir=root_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=num_workers)
    return train_loader, val_loader, test_loader
