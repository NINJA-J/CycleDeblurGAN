import torch.utils.data as data
import os
import random
import torch
import numpy as np
import torchvision

# from PIL import Image as Img
from MotionBlur.BlurKernel import BlurKernel
from torchvision import transforms
from PIL import ImageFile, Image
import PIL.PngImagePlugin

IMG_EXTENSIONS = [
    'jpg', 'JPG', 'jpeg', 'JPEG',
    'png', 'PNG', 'ppm', 'PPM', 'bmp', 'BMP',
]

ImageFile.LOAD_TRUNCATED_IMAGES = True

# class ImageDataSet(data.Dataset):

class ImageTestDataSet(data.Dataset):
    def __init__(self, iList, kList, size=20):
        pass

class ImageDataSet(data.Dataset):
    def __init__(self, dataFolder="Data", group='Disentangle', useTest=False, sImage=256, sKernel=64, rand=False, refine=None, size=1000):
        """

        :param dataFolder:
        :param kernelFolder:
        :param imgFolder:
        :param sImage:      Size of Image
        :param sKernel:     Size of Kernel
        """

        self.sImage = sImage
        self.sKernel = sKernel

        self.imgPaths = []
        self.krlPaths = []

        self.size = size
        self.rand = rand
        self.refine = refine
        self.group = group

        # print(os.path.join(dataFolder, 'Sharp'))
        if group == 'CelebA':
            imageFolder = os.path.join(dataFolder, 'Images', group)
        else:
            imageFolder = os.path.join(dataFolder, 'Images', group, 'Sharp')
        print("Image Folder: %s" % imageFolder)

        for root, _, files in os.walk(imageFolder):
            for file in files:
                if file.split(sep='.')[-1] in IMG_EXTENSIONS:
                    self.imgPaths.append(os.path.join(root, file))
        for root, _, files in os.walk(os.path.join(dataFolder, 'Kernel')):
            for file in files:
                if file.split(sep='.')[-1] == "bin":
                    self.krlPaths.append(os.path.join(root, file))

        self.imgs = len(self.imgPaths)
        self.krls = len(self.krlPaths)

        assert self.imgs > 0, "There are no Images at all"
        assert self.krls > 0, "There are no Kernels at all"

        tImageList = []
        tKernelList = []
        if useTest:
            for i in range(max(1, self.imgs//20)):
                tImageList.append(self.imgPaths.pop(random.randint(0, len(self.imgPaths))))
            for i in range(max(1, self.krls//20)):
                tKernelList.append(self.krlPaths.pop(random.randint(0, len(self.krlPaths))))
            self.imgs = len(self.imgPaths)
            self.krls = len(self.krlPaths)
            self.size -= self.size//20

        self.testDataSet = ImageTestDataSet(self, tImageList, tKernelList)

        class ToDouble(object):
            def __call__(self, input):
                # print(input)
                return input.double()

        # self.crop

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ToDouble(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
        ])
        # pass

    def __getitem__(self, item):
        def crop(img):
            if self.refine is not None and isinstance(self.refine, int):
                cropFn = transforms.Compose([
                    transforms.RandomCrop(size=self.sImage*self.refine),
                    transforms.Resize(size=self.sImage, interpolation=Image.BICUBIC)
                ])
            elif self.group == 'CelebA':
                # print(img.size)
                size = np.minimum(img.size[0], img.size[1])
                cropFn = transforms.Compose([
                    transforms.RandomCrop(size=size),
                    transforms.Resize(size=self.sImage, interpolation=Image.BICUBIC)
                ])
            else:
                cropFn = transforms.RandomCrop(size=self.sImage)
            return cropFn(img)

        norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5))
        ])

        kId = random.randint(0, self.krls - 1)
        iId = [
            random.randint(0, self.imgs - 1),
            random.randint(0, self.imgs - 1) if self.rand else item
        ]

        kernel = BlurKernel(self.sKernel)
        kernel.loadPSF(self.krlPaths[kId])

        sImage = [
            crop(Image.open(self.imgPaths[iId[0]])),
            crop(Image.open(self.imgPaths[iId[1]]))
        ]

        images = [
            norm(sImage[0]),
            norm(sImage[1]),
            norm(kernel.blur(sImage[0])),
            norm(kernel.blur(sImage[1]))
        ]

        # sImage, bImage = [], []
        #
        # for i in range(2):
        #     sImage.append(kernel.openImg(self.imgPaths[iId[i]]))
        #     sImage[i] = sImage[i].astype(np.double)
        #
        #     # print(sImage[i].shape)
        #     w = sImage[i].shape[2]
        #     h = sImage[i].shape[1]
        #     wOffset = random.randint(0, max(0, w - self.sImage - 1))
        #     hOffset = random.randint(0, max(0, h - self.sImage - 1))
        #
        #     # print(sImage[i].shape)
        #     sImage[i] = sImage[i][:, hOffset:hOffset + self.sImage, wOffset:wOffset + self.sImage]
        #     bImage.append(kernel.blur(sImage[i]))
        #
        #     # print(sImage[i].shape)
        #     # print(bImage[i].shape)
        #     # sImage[i] = self.transform(sImage[i].transpose((1, 2, 0)))
        #     # bImage[i] = self.transform(bImage[i].transpose((1, 2, 0)))

        bKernel = torch.Tensor(kernel.PSF)

        # images = sImage + bImage
        # for i in range(4):
        #     images[i] = transforms.ToTensor()(images[i].transpose((1, 2, 0))).double() / 255
        #     images[i] = (images[i] - 0.5) / 0.5
        #     # # t=torch.Tensor()
        #     # img = img.type(torch.double)
        #     # print(img.dtype)
        #     # img = transforms.Normalize(
        #     #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        #     # )(img)

        # print("Load Finished")

        return dict(
            sharp1=images[0],
            sharp2=images[1],
            blur1=images[2],
            blur2=images[3],
            kernel=bKernel
        )
        # pass
        #

    def __len__(self):
        # return self.krls * self.imgs
        if self.rand:
            return self.size
        else:
            return self.imgs

class ImageDataLoader(data.DataLoader):
    def __init__(self, dataSet, batchSize=1, shuffle=True, nWorkers=2):
        super(ImageDataLoader, self).__init__(dataSet, batch_size=batchSize, shuffle=shuffle, num_workers=nWorkers)
        self.dataSet = dataSet
        # pass

    def __len__(self):
        return len(self.dataSet)
