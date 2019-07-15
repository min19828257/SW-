import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, Resize

from os import listdir
from os.path import join
from PIL import Image

import cv2
import random
NUM_EPOCHS = 1
loadmodel = None

GPUUSE = None


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        return F.sigmoid(self.net(x))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.conv4 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


# save load 코드 예시
# torch.save(netD.state_dict(), 'param/netD_epoch_%d.pth' % (epoch))

# net.load_state_dict(torch.load(loadmodel))

class DatasetFromfolder(Dataset):
    def __init__(self, dir_mnist):
        super(DatasetFromfolder, self).__init__()
        #dir_mnist = 'C:\Users\Administrator\Desktop\새 폴더\SW-\7월15일 generator\4주차\mnist_png.tar\mnist_png\training'
        self.filelist = []
        self.lensum = 0
        for i in range(10):
            idir = join(dir_mnist, str(i))
            filelist_tmp = [join(idir, x) for x in listdir(idir) if is_image_file(x)]
            self.filelist.append((filelist_tmp, i))
            self.lensum = self.lensum + len(filelist_tmp)

        self.transform = Compose([ToTensor()])

    def __getitem__(self, index):
        c = random.randint(10)
        clist, label = self.filelist[c]
        resultimage = self.transform(Image.open(clist[index]).convert('L'))

        return resultimage, label

    def __len__(self):
        return  self.lensum


def train():
    train_set = DatasetFromfolder('./data/1', './data/0')
    val_set = DatasetFromfolder('./data1/1', './data1/0')
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=True)

    GPUUSE = None

    # netD = Discriminator()
    netD = Net()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    criterion = nn.MSELoss()
    if loadmodel is not None:
        print("=> loading checkpoint '{}'".format(loadmodel))
        netD.load_state_dict(torch.load(loadmodel))
    if GPUUSE == None:
        netD.cpu()
    else:
        if torch.cuda.is_available():
            netD.cuda()

    optimizerD = optim.Adam(netD.parameters())

    for epoch in range(1, NUM_EPOCHS + 1):

        netD.train()
        batch_idx = 0
        for sample, label in train_loader:
            batch_size = sample.size(0)

            if GPUUSE == None:
                face = sample.cpu()
                label = label.cpu()
            else:
                if torch.cuda.is_available():
                    face = sample.cuda()
                    label = label.cuda()

            netD.zero_grad()
            out = netD(face)

            d_loss = criterion(out.squeeze(), label.squeeze())
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            with torch.no_grad():
                for val_face, val_label in val_loader:
                    val_out = netD(val_face)
                    Val_loss = criterion(val_out, val_label)
                    print('Epoch [{}/{}], BatchStep[{}/{}], Loss: {}'.format(epoch, NUM_EPOCHS, batch_idx, batch_size,
                                                                             Val_loss))
            batch_idx += 1

        torch.save(netD.state_dict(), 'param/netD_epoch_%d.pth' % (epoch))


def test(img):
    print("test session")

    loadmodel = 'param/netD_epoch_1.pth'
    # netD = Discriminator()
    net = Net()
    print('# discriminator parameters:', sum(param.numel() for param in net.parameters()))

    if GPUUSE == None:
        net.cpu()
    else:
        if torch.cuda.is_available():
            net.cuda()

    if loadmodel is not None:
        print("=> loading checkpoint '{}'".format(loadmodel))
        net.load_state_dict(torch.load(loadmodel))

    net.eval()

    trans = Compose([Resize([144, 144]), ToTensor()])
    img.convert("L")

    img = trans(img).unsqueeze(1)
    if GPUUSE != None:
        img = img.cuda()

    output = net(img)

    print("output : ", output[0])

    return output[0]


if __name__ == "__main__":
    # train()
    test()