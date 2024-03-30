import torch

def save_model(PATH, model):
    torch.save(model, PATH)

def read_model(PATH):
    return torch.load(PATH)

def save_checkpoint(PATH, checkpoint_info):
    torch.save(checkpoint_info, PATH)
    
def read_checkpoint(PATH):
    return torch.load(PATH)