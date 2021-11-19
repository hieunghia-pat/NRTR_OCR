from torch.nn import functional as F 
from torch import nn 

class ModalityTransformer(nn.Module):
    def __init__(self, imChannel, imgHeight, d_model):
        super(ModalityTransformer, self).__init__()

        self.conv1 = nn.Conv2d(imChannel, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = F.relu

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = F.relu

        self.fc = nn.Linear(64 * (imgHeight // 4), d_model)

    def forward(self, x):
        x = self.bn1(self.relu1(self.conv1(x)))
        x = self.bn2(self.relu2(self.conv2(x)))
        bs, c, h, w = x.shape
        x = x.contiguous().view((bs, c*h, w)).permute((0, 2, 1))
        x = self.fc(x)

        return x