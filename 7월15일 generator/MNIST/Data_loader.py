from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os.path import join
from os import listdir


class DatasetFromfolder(Dataset):
    def __init__(self, dir_mnist):
        super(DatasetFromfolder, self).__init__()

        self.filelist = []
        self.lenlist = []
        self.lensum = 0
        for i in range(10):
            idir = join(dir_mnist, str(i))
            filelist_tmp = [join(idir, x) for x in listdir(idir)]
            self.filelist.append((filelist_tmp, i))
            self.lenlist.append(len(filelist_tmp))
            self.lensum = self.lensum + len(filelist_tmp)

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):

        c, cindex = self._findlabelfromindex(index)
        clist, label = self.filelist[c]
        resultimage = self.transform(Image.open(clist[cindex]).convert('L'))

        return resultimage, label

    def __len__(self):
        return self.lensum

    def _findlabelfromindex(self, index):
        label = 0
        indexsum = 0

        for i in range(10):
            indexsum += self.lenlist[i]
            if index < indexsum:
                label = i
                break

        classindex = index - indexsum


        return label, classindex

