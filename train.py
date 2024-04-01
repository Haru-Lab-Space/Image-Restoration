import torch
import torch.nn as nn
from torch.optim import Adam
from torch import optim
from tqdm import tqdm
from utils.loss_func import ssim
import os
from utils.file_interaction import mkdir
from config import load_config
from utils.scheduler import GradualWarmupScheduler, CosineDecayWithWarmUpScheduler
from models.model_1 import Model_1
from torch.utils.data import DataLoader


class Trainer():
    def __init__(self, args, train_dataset, valid_dataset, loss_func):
        """
            model_param trong init model là các đối số để khởi tao model
        """
        # self.args = load_config()
        self.args = args
        # cái này cần được bổ sung thêm
        self.model = Model_1()
        self.loss_func = loss_func
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def init_hyperparams(self):
        if self.args['save_checkpoint'] is None:
            mkdir('save_checkpoint')
        else:
            mkdir(self.args['save_checkpoint'])
        if self.args['load_checkpoint'] is not None:
            checkpoint = torch.load(self.args['load_checkpoint'])
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to('cuda')

        # learning rate
        new_lr = self.args['learning_rate']

        # optimizer
        optimizer = Adam(self.model.parameters(), lr=new_lr)

        # scheduler
        total_iters = self.args['num_epochs']
        start_iter = 1
        warmup_iter = 50
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, total_iters-warmup_iter, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine)
        scheduler.step()

        # data loader
        train_loader = DataLoader(
            self.train_dataset, self.args['batch_size'], shuffle=True)
        valid_loader = DataLoader(
            self.valid_dataset, self.args['batch_size'], shuffle=True)

        return optimizer, new_lr, scheduler, train_loader, valid_loader

    def valid(self, valid_loader, epoch):
        self.model.eval()
        loss_sum = 0
        num_iterations = 0
        # losses_valid = []
        data_iterator = tqdm(valid_loader, desc=f'Epoch {epoch+1}')
        for data in tqdm(data_iterator):
            num_iterations += 1

            input_video, target = data['input'], data['output']
            input_video = input_video.to('cuda')
            input_video = input_video.permute(0, 2, 1, 3, 4)
            target = target.to('cuda')
            target = target.permute(0, 2, 1, 3, 4)
            output = self.model(input_video)
            # compute loss
            loss = self.loss_func(output, target)
            # print(type(loss))
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            # losses_valid.append(avg_loss)
            data_iterator.set_postfix(loss=avg_loss)

    def train(self, train_loader, optimizer, scheduler, epoch):
        self.model.train()

        loss_sum = 0
        num_iterations = 0
        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for data in data_iterator:
            num_iterations += 1
            input_video, target = data['input'], data['output']
            # print(input_video)
            input_video = input_video.to('cuda')
            input_video = input_video.permute(0, 2, 1, 3, 4)
            target = target.to('cuda')
            target = target.permute(0, 2, 1, 3, 4)
            # output model
            output = self.model(input_video)

            optimizer.zero_grad()
            # compute loss
            loss = self.loss_func(output, target)
            # print(type(loss))
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            # losses.append(avg_loss)
            data_iterator.set_postfix(loss=avg_loss)

            loss.backward()
            optimizer.step()
            scheduler.step()

        torch.save({'iter': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join('save_checkpoint', f"model_{epoch}.pth"))

    def start_training(self):
        optimizer, new_lr, scheduler, train_loader, valid_loader = self.init_hyperparams()
        print("Start Traning ")
        for epoch in range(self.args['num_epochs']):
            self.train(train_loader=train_loader,
                       optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            self.valid(valid_loader=valid_loader, epoch=epoch)
        print("Done")
