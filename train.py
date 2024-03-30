import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from utils.loss_func import ssim
import os

device = 'cpu' if torch.cuda.is_available() else 'cuda'


def init_model(model=None):
    pass


def train(args):
    if args.save_checkpoint is None:
        os.mkdir('save_checkpoint')

    # load dataset and dataLoader
    train_dataset = None
    train_loader = None

    val_dataset = None
    val_loader = None

    # Init parameter of model before training
    model = init_model()
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    new_lr = args.learning_rate

    # optimizer
    optimizer = Adam(model.parameters(), lr=new_lr)

    # schedule

    # trainingS
    total_iterations = 0
    losses = []
    model.train()
    print("Start Training")
    for epoch in range(args.num_epochs):
        loss_sum = 0
        num_iterations = 0
        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            input, target = data
            input = input.to(device)
            target = target.to(device)
            output = model(input)

            optimizer.zero_grad()

            loss = ssim(output, target)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            losses.append(avg_loss)
            data_iterator.set_postfix(loss=avg_loss)

            loss.backward()
            optimizer.step()

        torch.save({'iter': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join('save_checkpoint', f"model_{epoch}.pth"))
