from torch import nn

class Discriminator_FCN(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.FCN1 = nn.Linear(784, 100)
        self.FCN2 = nn.Linear(100, 1)
        self.sig = nn.Sigmoid()
        self.LRU = nn.LeakyReLU(0.2)
    def forward(self, z):
        out1 = self.FCN1(z)
        out3 = self.LRU(out1)
        out4 = self.FCN2(out3)
        out5 = self.LRU(out4)
        result = self.sig(out5)
        return result

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Conv1 = nn.Conv2d(1, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.Conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(256)
        self.Conv3 = nn.Conv2d(256, 512, 4, 1, 0)
        self.bn3 = nn.BatchNorm2d(512)
        self.Conv4 = nn.Conv2d(512, 1, 4, 1, 0)
        self.LRU = nn.LeakyReLU(0.2)
        self.sig = nn.Sigmoid()

    def forward(self, z):
        z = self.LRU(self.bn1(self.Conv1(z)))
        z = self.LRU(self.bn2(self.Conv2(z)))
        z = self.LRU(self.bn3(self.Conv3(z)))
        out = self.sig(self.LRU(self.Conv4(z)))


        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.DeConv1 = nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.DeConv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(256)
        self.DeConv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.DeConv4 = nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        self.Relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):

        out1 = self.Relu(self.bn1(self.DeConv1(z)))
        out2 = self.Relu(self.bn2(self.DeConv2(out1)))
        out3 = self.Relu(self.bn3(self.DeConv3(out2)))
        out4 = self.tanh(self.DeConv4(out3))

        return out4
