import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from utils.loss_func import ssim
import os
from utils.file_interaction import mkdir
from config import load_config
from utils.scheduler import WarmupCosineScheduleRestarts, GradualWarmupScheduler

# get device
device = 'cpu' if torch.cuda.is_available() else 'cuda'


class Trainer():
    def __init__(self, args, train_dataset, valid_dataset, model_param, loss_func):
        """
            model_param trong init model là các đối số để khởi tao model
        """
        self.args = load_config()
        # cái này cần được bổ sung thêm
        self.model = self.init_model(model_param)
        self.loss = loss_func()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def init_model(model_param):
        pass

    def init_hyperparams(self):
        if self.args.save_checkpoint is None:
            mkdir(self.args.save_checkpoint)
        if self.args.load_checkpoint is not None:
            checkpoint = torch.load(self.args.load_checkpoint)
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)

        # learning rate
        new_lr = self.args.learning_rate

        # optimizer
        optimizer = Adam(self.model.parameters(), lr=new_lr)

        # scheduler
        total_iters = self.args.num_epochs
        start_iter = 1
        warmup_iter = 5000
        scheduler_cosine = nn.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, total_iters-warmup_iter, eta_min=1e-6)
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1, total_epoch=warmup_iter, after_scheduler=scheduler_cosine)
        scheduler.step()

        # data loader
        train_loader = nn.utils.DataLoader(
            self.train_dataset, self.args.batch_size, shuffle=True)
        valid_loader = nn.utils.DataLoader(
            self.valid_dataset, self.args.batch_size, shuffle=True)

        return optimizer, new_lr, scheduler, train_loader, valid_loader

    def valid(self, valid_loader, epoch):
        self.model.eval()
        loss_sum = 0
        num_iterations = 0
        # losses_valid = []
        data_iterator = tqdm(valid_loader, desc=f'Epoch {epoch+1}')
        for data in tqdm(data_iterator):
            num_iterations += 1
            total_iterations += 1

            input_, target = data
            input_ = input_.to(device)
            target = target.to(device)

            output = self.model(input_)
            # compute loss
            loss = self.loss_func(output, target)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            # losses_valid.append(avg_loss)
            data_iterator.set_postfix(loss=avg_loss)

    def train(self, train_loader, optimizer, scheduler, epoch):
        # if self.args.save_checkpoint is None:
        #     mkdir(self.args.save_checkpoint)

        # load dataset and dataLoader
        # train_loader = None

        # # Init parameter of model before training
        # # model = init_model()
        # if self.args.load_checkpoint is not None:
        #     checkpoint = torch.load(self.args.load_checkpoint)
        #     self.model.load_state_dict(checkpoint['state_dict'])
        # self.model.to(device)
        # new_lr = self.args.learning_rate

        # # optimizer
        # optimizer = Adam(self.model.parameters(), lr=new_lr)

        # # schedule
        # trainingS
        self.model.train()

        loss_sum = 0
        num_iterations = 0
        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            input_, target = data
            input_ = input_.to(device)
            target = target.to(device)
            output = self.model(input_)

            optimizer.zero_grad()
            # compute loss
            loss = self.loss_func(output, target)
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
        for epoch in range(self.args.num_epochs):
            self.train(train_loader=train_loader,
                       optimizer=optimizer, scheduler=scheduler, epoch=epoch)
            self.valid(valid_loader=valid_loader, epoch=epoch)
        print("Done")
