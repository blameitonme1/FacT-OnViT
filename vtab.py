import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision import transforms
import torch
_DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)
# 各个数据集每一个的分类数量是多少
_CLASSES_NUM = (100, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9)

def get_classes_num(dataset_name):
    dict_ = {name : num for name, num in zip(_DATASET_NAME, _CLASSES_NUM)}
    return dict_[dataset_name]

def default_loader(path):
    # 加载图像
    return Image.open(path).convert('RGB')

def default_flist_reader(flist):
    """
    flist的format是 impath label\nimpath label\n ...
    上面简单来说就是每一行都是Image path加上对应label, 最后返回一个list
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))
    
    return imlist

class ImageFilelist(data.Dataset):
    """ 数据集的类 """
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        super().__init__()
        self.root = root # 图片数据集根目录，毕竟impath是相对路径
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    def __getitem__(self, index):
        # target就是label
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath).replace('/', '\\'))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    def __len__(self):
        return len(self.imlist)

def get_data(name, evaluate=True, batch_size=16):
    # 得到某一个name指定的数据集的数据
    root = './data/caltech101/' + name
    # interpolation指定如何插值填充未知位置，注意compose参数是一个list
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    if evaluate:
        # 预测
        train_loader = data.DataLoader(
            ImageFilelist(root=root, flist=root + "/train800val200.txt",
                          transform=transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=1, pin_memory=True
        )
        # 这里batch_size设置更大因为验证集更少，并且不需要更新模型
        val_loader = data.DataLoader(
            ImageFilelist(root=root, flist=root + "/test.txt",
                          transform=transform),
            batch_size=16, shuffle=False,
            num_workers=1, pin_memory=True
        )
    else:
        # 训练，使用验证集
        train_loader = data.DataLoader(
            ImageFilelist(root=root, flist=root + "/train800.txt",
                          transform=transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=1, pin_memory=True
        )
        val_loader = data.DataLoader(
            ImageFilelist(root=root, flist=root + "/val200.txt",
                          transform=transform),
            batch_size=16, shuffle=False,
            num_workers=1, pin_memory=True
        )
    return train_loader, val_loader



