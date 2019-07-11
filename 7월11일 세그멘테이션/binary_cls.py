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
GPUUSE = True

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['bmp', '.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        ## 네트워크 init
        #
        #
        ################


    def forward(self, x):

        ## forward
        #
        #
        #################

class DatasetFromfolder(Dataset):
    def __init__(self, facedataset_dir, nonfacedataset_dir):
        super(DatasetFromfolder, self).__init__()
        #####
        #
        #
        #####


    def __getitem__(self, index):

        #####
        #
        #
        #####

        #return resultimage, label

    def __len__(self):
        return ######

def train():

    train_set = #
    val_set = #
    train_loader = #
    val_loader = #


    # 네트워크 불러오기
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # parameter가 있다면 불러오는 부분

    # loss 와 optimizer 정의

    # 실제 train이 돌아가는 for문:
        #dataloader에서 input을 받고 학습시키는 부분
        #필요하다면 validation set으로 loss 확인
        #매 epoch 마다 parameter를 저장하기


def test():

    # 네트워크 불러오기

    # parameter 받아오기

    # test할 data 받아오기 (인터넷에서 아무거나 받아와서 확인해봐도 됨!)

    # 네트워크 통과시켜서 결과 display

if __name__ == "__main__":
    train()
    #test()