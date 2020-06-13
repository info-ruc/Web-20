import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, cls_num=50, x_dim=3, hid_dim=64, z_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim*4, cls_num)
        self.rotFc = nn.Linear(z_dim, 4)


    def forward(self, x):
        x = x.view(-1,3,84,84)
        x = self.encoder(x)
        x = nn.MaxPool2d(5)(x)
        x = x.view(x.size(0), -1)
        
        logits = self.fc(x.view(-1, self.z_dim*4))
        rotLogits = self.rotFc(x)
        return logits, rotLogits
