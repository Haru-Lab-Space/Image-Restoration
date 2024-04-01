import torch


def save_model(PATH, model):
    torch.save(model, PATH)


def read_model(PATH):
    return torch.load(PATH)


def save_checkpoint(PATH, checkpoint_info):
    torch.save(checkpoint_info, PATH)


def init_model(model_params):
    return model_params


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['iter']
    if optimizer is not None:
        return model, optimizer, epoch
    else:
        return model, epoch
