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

NUM_EPOCHS = 5
loadmodel = 'param/netD_epoch_5.pth'

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
#torch.save(netD.state_dict(), 'param/netD_epoch_%d.pth' % (epoch))

#net.load_state_dict(torch.load(loadmodel))

class DatasetFromfolder(Dataset):
    def __init__(self, facedataset_dir, nonfacedataset_dir):
        super(DatasetFromfolder, self).__init__()
        self.face_image_filenames = [join(facedataset_dir, x) for x in listdir(facedataset_dir) if is_image_file(x)]
        self.nonface_image_filenames = [join(nonfacedataset_dir, x) for x in listdir(nonfacedataset_dir) if is_image_file(x)]

        self.face_transform = Compose([ToTensor()])
        self.nonface_transform = Compose([RandomCrop(144),ToTensor()])


    def __getitem__(self, index):
        resultimage = 0
        label = 0
        if index < len(self.face_image_filenames):
            resultimage = self.face_transform(Image.open(self.face_image_filenames[index]).convert('L'))
            label = torch.ones(1)
        else:
            resultimage = self.nonface_transform(Image.open(self.nonface_image_filenames[index - len(self.face_image_filenames)]))
            label = torch.zeros(1)

        return resultimage, label

    def __len__(self):
        return len(self.face_image_filenames) + len(self.nonface_image_filenames)



def train():
    train_set = DatasetFromfolder('data/binary_cls/face', 'data/binary_cls/nonface')
    val_set = DatasetFromfolder('data/binary_cls/val_face', 'data/binary_cls/val_nonface')
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
                    # if torch.cuda.is_available():
                    #     val_face = val_face.cuda()
                    #     val_label = val_label.cuda()

                    val_out = netD(val_face)

                    Val_loss = criterion(val_out, val_label)

                    print('Epoch [{}/{}], BatchStep[{}/{}], Loss: {}'.format(epoch, NUM_EPOCHS, batch_idx, batch_size, Val_loss))

            batch_idx += 1

        torch.save(netD.state_dict(), 'param/netD_epoch_%d.pth' % (epoch))


def test():
    print("test session")

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

    img_name = "IU1.png"
    img = Image.open(img_name).convert('L')
    # img = Image.open("data/binary_cls/IU2.png").convert('L')
    trans = Compose([Resize([144,144]),ToTensor()])

    img = trans(img).unsqueeze(1)
    if GPUUSE != None:
        img = img.cuda()

    ###cv2 로도 되나
    # im = cv2.imread("data/binary_cls/IU2.png", 0).astype('float32')
    # im = cv2.resize(im, (144, 144)) / 256
    # im = torch.from_numpy(im)
    # im = im.unsqueeze(0).unsqueeze(0).cuda()
    #################

    output = net(img)

    im = cv2.imread(img_name)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, "%0.4f"%float(output), (10, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("im",im)
    cv2.waitKey()

    print(output)

    # img = Image.open("data/binary_cls/val_nonface/5.jpg")
    # trans = Compose([RandomCrop(144),ToTensor()])
    # img = trans(img).unsqueeze(1)
    # if GPUUSE != None:
    #     img = img.cuda()
    #
    # output = net(img)
    # print(output)

if __name__ == "__main__":
    train()
    # test()