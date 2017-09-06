"""Trainer that will import required model, run whole training and monitor
the process"""
import time
import os
import getpass
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

from lr_scheduler import MultiStepLR
from data_loader import get_train_val_test_dataset
# from models import model_conv as model
from models import model_dense as model
from monitoring import FoldersManager, MainLog


class Trainer():
    def __init__(self, model, criterion, optimizer_params,
                 fold_man, monitor,
                 data_loaders, use_cuda, gpu_idx=0):

        if criterion == 'cross_enthropy':
            self.criterion = nn.CrossEntropyLoss()
        # define optimizer
        if optimizer_params['type'].lower() == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=optimizer_params['lr'],
                momentum=optimizer_params.get('momentum', 0),
                weight_decay=optimizer_params.get('weight_decay', 0),
                nesterov=optimizer_params.get('nesterov', False))
        elif optimizer_params['type'].lower() == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=optimizer_params['lr'],
                weight_decay=optimizer_params.get('weight_decay', 0))
        # define learning rate scheduler
        if optimizer_params.get('lr_scheduler'):
            self.lr_scheduler = MultiStepLR(
                self.optimizer,
                milestones=optimizer_params['lr_scheduler']['milestones'],
                gamma=optimizer_params['lr_scheduler']['gamma'])
        else:
            self.lr_scheduler = None

        self.fold_man = fold_man
        self.monitor = monitor
        self.use_cuda = use_cuda
        self.gpu_idx = gpu_idx
        self.model = self.to_gpu(model)
        self.data_loaders = data_loaders
        self.best_acc = 0.0
        self.best_model_path = str(
            self.fold_man.model_run_saves_dir / 'best_model.chkpt')

    def to_gpu(self, tensor):
        if self.use_cuda:
            return tensor.cuda(self.gpu_idx)
        else:
            return tensor

    def from_gpu(self, tensor):
        if self.use_cuda:
            return tensor.cpu()
        else:
            return tensor

    def train_val_one_epoch(self, data_loader, epoch, train=True):
        train_str = 'train' if train else 'val'
        self.model.train(train)
        if self.lr_scheduler:
            self.lr_scheduler.step()
        start_time = time.time()
        running_loss_epoch = []
        correct = 0
        total = 0
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data
            inputs = Variable(self.to_gpu(inputs))
            labels = Variable(self.to_gpu(labels))

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            if train:
                # compute gradient with respect to loss
                loss.backward()
                # apply gradients
                self.optimizer.step()

            _, pred = torch.max(outputs.data, 1)

            correct += (pred == labels.data).sum()
            total += labels.size(0)
            running_loss_epoch.append(loss.data[0])
            self.monitor[train_str].log('batch_loss', loss.data[0])

        mean_loss = np.mean(running_loss_epoch)
        self.monitor[train_str].log('epoch_loss', mean_loss)
        acc = correct / total
        self.monitor[train_str].log('accuracy', acc)
        time_cons = time.time() - start_time
        str_time = str(timedelta(seconds=time_cons))
        msg = "Epoch: {}, loss: {}, acc: {}, time per epoch: {}"
        print(msg.format(epoch, mean_loss, acc, str_time))
        if not train:
            self.save_best_model(acc)

    def save_best_model(self, acc):
        if acc > self.best_acc:
            print("best model were saved")
            print(self.best_model_path)
            self.best_acc = acc
            self.model.save(self.best_model_path)

    def train_val_all_epochs(self, epochs):
        self.losses = {'train': [], 'val': []}
        self.accuracies = {'train': [], 'val': []}
        for epoch in range(epochs):
            print('-' * 10)
            self.train_val_one_epoch(
                self.data_loaders['train'], epoch, train=True)
            self.train_val_one_epoch(
                self.data_loaders['val'], epoch, train=False)
            self.monitor.dump_logs(
                self.fold_man.model_run_logs_dir / 'logs.json')

    def test_best_model(self):
        print("Evaluate best model")
        self.model.load(self.best_model_path)
        self.model.train(False)
        test_predictions = []
        for inputs, names in test_loader:
            img_name = os.path.basename(names[0])
            inputs = Variable(self.to_gpu(inputs))

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            _, pred = torch.max(outputs.data, 1)
            pred = self.from_gpu(pred).numpy()[0][0]
            class_ = int(pred) + 1
            test_predictions.append({
                'image': img_name,
                'class': class_
            })
        df = pd.DataFrame.from_dict(test_predictions)
        df.to_csv(str(self.fold_man.model_run_pred_dir / 'pred.csv'),
                  index=False, header=True)


if getpass.getuser() == 'illarionkhliestov':
    use_cuda = False
    root_dir = '/Users/illarionkhliestov/Datasets/caltech-256/'
    loader_workers = 4
else:
    use_cuda = True
    root_dir = '/home/illarion/datasets/caltech-256/'
    loader_workers = 4

### Get loader
train_csv = os.path.join(root_dir, 'df_train_split.csv')
val_csv = os.path.join(root_dir, 'df_val_split.csv')
test_csv = os.path.join(root_dir, 'test.csv')

conv_optimizer = {
    'type': 'adam',
    'lr': 42,
}

dense_optim = {
    'type': 'SGD',
    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'nesterov': True,
    'lr_scheduler': {
        'milestones': [150, 225],
        'gamma': 0.1,
    }
}

model_params = {
    'epochs': 300,
    'dropout_rate': 0.2,
    'criterion': 'cross_enthropy',
    'optimizer_params': dense_optim,
    'n_layers': 4,
}

data_params = {
    'resize': (64, 64),
    'batch_size': 64,
    'augmentation': [
        'transforms.RandomCrop((64, 64), padding=4)',
        'transforms.RandomHorizontalFlip()',
    ]
}

train_loader, val_loader, test_loader = get_train_val_test_dataset(
    train_csv, val_csv, test_csv, root_dir=root_dir,
    batch_size=data_params['batch_size'])
data_loaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader,
}
model = model.Model(**model_params)

fold_man = FoldersManager(
    model_graph_decl=str(model),
    logs_dir='logs',
    saves_dir='saves',
    pred_dir='predictions',
    train_params=model_params,
    data_params=data_params,
    test_mode=False)
monitor = MainLog(logged_states=['train', 'val'])

# get required instances
trainer = Trainer(
    model=model,
    criterion=model_params['criterion'],
    optimizer_params=model_params['optimizer_params'],
    fold_man=fold_man,
    monitor=monitor,
    use_cuda=use_cuda,
    data_loaders=data_loaders)
try:
    trainer.train_val_all_epochs(model_params['epochs'])
except KeyboardInterrupt:
    pass
trainer.test_best_model()
