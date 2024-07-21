import os
import torch
import torch.nn as nn


def modelSave(model, save_path, save_name):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


def modelLoad(model, model_path, data_parallel=False):
    model.load_state_dict(torch.load(model_path), strict=True)
    if data_parallel:
        model = nn.DataParallel(model)
    return model