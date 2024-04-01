from torch import nn


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv_layer = nn.Conv3d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        return self.conv_layer(x)
