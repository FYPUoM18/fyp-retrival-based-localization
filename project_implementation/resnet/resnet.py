import torch

class resent_model:
    model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
