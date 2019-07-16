import torch
import torch.nn as nn
from model import Generator, Discriminator
from Data_loader import DatasetFromfolder
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

NUM_EPOCHS = 100
def train():
    train_set = DatasetFromfolder('C:/Users/Administrator/Downloads/mnist_png-master/mnist_png-master/mnist_png.tar/mnist_png/training')
    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=16, shuffle=True)

    netG = Generator()
    netD = Discriminator()

    criterion = nn.MSELoss()

    optimizerD = optim.Adam(netD.parameters())
    optimizerG = optim.Adam(netG.parameters())

    for epoch in range(1, NUM_EPOCHS + 1):
        batch_idx = 0
        for x, label in train_loader:

            batch_size = x.size(0)

            x = 2*x - 1
            z = torch.rand(batch_size, 100, 1, 1)
            fake_image = netG(z)

            fake = netD(fake_image)
            real = netD(x)

            netD.zero_grad()

            d_loss = criterion(fake.squeeze(), torch.zeros(batch_size)) + criterion(real.squeeze(), torch.ones(batch_size))
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            netG.train()
            netG.zero_grad()

            g_loss = criterion(fake.squeeze(), torch.ones(batch_size))
            g_loss.backward(retain_graph=True)
            optimizerG.step()

            if batch_idx % 20 == 0:
                netG.eval()

                eval_z = torch.rand(batch_size, 100, 1, 1)
                generated_image = netG(eval_z)

                generated_image = (generated_image + 1) / 2

                print("Epoch:{} batch[{}/{}] G_loss:{} D_loss:{}".format(epoch, batch_idx, len(train_loader), g_loss, d_loss))
                torchvision.utils.save_image(generated_image.data, 'result/Generated-%d-%d.png' % (batch_idx, epoch))

            batch_idx += 1


if __name__ == "__main__":
    train()
    # test()


