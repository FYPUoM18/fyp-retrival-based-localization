import torch.nn as nn

class ContrastiveModel(nn.Module):
    
    def __init__(self,conf) -> None:
        super(ContrastiveModel,self).__init__()
        self.conf=conf
        self.model =nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conf.input_shape[1]*self.conf.input_shape[2], 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Flatten(),
        )
    
    def forward(self,batch_as_tensor):
        return self.model(batch_as_tensor)
