import torch.utils.data as data
import os
import random
import numpy as np
from torchvision import transforms
from PIL import Image
from MotionBlur.BlurKernel import BlurKernel

class DisentangleDataSet(data.Dataset):
    def __init__(self, fTrain='./Data/Disentangle', sImage=512, size=1000):
        self.fTrain = fTrain
        self.sImage = sImage
        self.size = size

        self.sImgPaths, self.bImgPaths, self.krnlPaths = [], [], []

        for root, _, files in os.walk(os.path.join(fTrain, 'Sharp')):
            for file in files:
                if file.split(sep='.')[-1] == 'png':
                    self.sImgPaths.append(os.path.join(root, file))
        for root, _, files in os.walk(os.path.join(fTrain, 'Blur')):
            for file in files:
                if file.split(sep='.')[-1] == 'png':
                    self.bImgPaths.append(os.path.join(root, file))
        for root, _, files in os.walk(os.path.join(fTrain, 'Kernel')):
            for file in files:
                if file.split(sep='.')[-1] == 'bin':
                    self.krnlPaths.append(os.path.join(root, file))

        self.sImgCnt = len(self.sImgPaths)
        self.bImgCnt = len(self.bImgPaths)
        self.krnlCnt = len(self.krnlPaths)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        sId = random.randint(0, self.sImgCnt - 1)
        bId = random.randint(0, self.sImgCnt - 1)
        kId = random.randint(0, self.krnlCnt - 1)

        kernel = BlurKernel(canvas=64)
        kernel.loadPSF(self.krnlPaths[kId])

        sImage = kernel.openImg(self.sImgPaths[sId])
        bImage = kernel.openImg(self.sImgPaths[bId])

        image = [sImage, bImage]
        # print(sImage.shape, " ", bImage.shape
        for i in range(2):
            w = image[i].shape[2]
            h = image[i].shape[1]
            wOffset = random.randint(0, max(0, w - self.sImage - 1))
            hOffset = random.randint(0, max(0, h - self.sImage - 1))
            image[i] = image[i][:, hOffset:hOffset + self.sImage, wOffset:wOffset + self.sImage]  # / 255

        image[1] = kernel.blur(image[1])

        return dict(
            sharp=self.transform(image[0].transpose((1, 2, 0))),
            blur=self.transform(image[1].transpose((1, 2, 0)))
        )

    def __len__(self):
        return self.size


