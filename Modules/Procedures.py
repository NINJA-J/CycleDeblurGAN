# from PIL import Image as Img
# import numpy as np
import torch
# from torch import nn
from torchvision import transforms as trsf
from .Loss import PerceptualLoss
# from torch.autograd import Variable
# from torch.cuda import FloatTensor
# import torch.cuda.FloatTensor
import visdom as vis
# import random
import torch.nn as nn
import os
import numpy as np


class BlurKernelEstimator(nn.Module):
    def __init__(self, sPhoto=256, sKernel=64, batchSize=1):
        super(BlurKernelEstimator, self).__init__()

        self.sPhoto = sPhoto
        self.sKernel = sKernel

        self.lossFn1 = nn.MSELoss()
        self.lossFn2 = PerceptualLoss(nn.MSELoss())
        self.loss = None

        self.useGPU = True
        # self.useGPU = False

        self.iImage = None
        self.iKernel = None
        self.oKernel = None

        # self.vis = vis.Visdom(env=self.__class__.__name__)

        self.transform = trsf.Compose([
            trsf.ToTensor(),
            # TODO：考虑Normalize是否必要
            trsf.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            )
        ])


        self.model = torch.nn.Sequential(*[
            # 256 ==> 256
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, True),

            # 256 ==> 128
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, True),

            # 128 ==> 128
            nn.ReplicationPad2d(1),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),

            # 128 ==> 64
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, True),

            #

            nn.ReplicationPad2d(1),
            nn.Conv2d(32, 1, kernel_size=3),
            nn.Sigmoid(),
            nn.ReLU()
        ])

        self.optimizer = torch.optim.Adam(self.model.parameters())
        # model =

        # self.loss = nn.MSELoss()
        if self.useGPU:
            self.model = self.model.cuda()

    def forward(self, input):
        self.iImage = input['bImage']
        self.iKernel = input['bKernel']

        if self.useGPU:
            self.iImage = self.iImage.cuda()
            self.iKernel = self.iKernel.cuda()
            # print(self.iKernel)
            # print(self.model)
            # self.oKernel = nn.parallel.data_parallel(self.model, self.iImage, [0])
            self.oKernel = self.model.forward(self.iImage)
        else:
            self.oKernel = self.model.forward(self.iImage)

        return self.oKernel

    def backward(self):
        self.optimizer.zero_grad()

        # iKernel = self.iKernel.reshape((1,) + self.iKernel.shape)
        # oKernel = self.iKernel.reshape((1,) + self.oKernel.shape)

        # self.loss = \
        #     self.lossFn1(self.iKernel, self.oKernel) + \
        #     self.lossFn2(iKernel, oKernel)
        self.loss = nn.MSELoss()(self.iKernel, self.oKernel)
        # self.loss = nn.KLDivLoss()(self.iKernel, self.oKernel)
        self.loss.backward()

        self.optimizer.step()

    def currError(self):
        return dict(
            loss=self.loss.item()
        )

    def currData(self):

        iImage, iKernel, oKernel =\
            self.iImage[0].cpu().float().numpy(),\
            self.iKernel[0].cpu().float().numpy(), \
            self.oKernel[0, 0].cpu().float().detach().numpy()
        # print(iImage.shape)
        iImage = (iImage + 1) / 2.0 * 255.0
        # iImage = iImage.transpose((2, 0, 1))
        iKernel = iKernel / np.max(iKernel) * 255
        oKernel = oKernel / np.max(oKernel) * 255

        return dict(
            iImage=iImage,
            iKernel=iKernel,  # tensor2im(self.iKernel),
            oKernel=oKernel  # tensor2im(self.oKernel)
        )

    def save(self, name='model'):
        torch.save(self.model.cpu().state_dict(), "CheckPoints\\%s.pth" % name)
        if self.useGPU:
            self.model = self.model.cuda()

    def load(self, name='model'):
        fileName = "CheckPoints\\%s.pth" % name
        if os.path.exists(fileName):
            self.model.load_state_dict(torch.load("CheckPoints\\%s.pth" % name))
            print("Model Loaded from %s" % fileName)
        if self.useGPU:
            self.model.cuda()



if __name__=='__main__':
    factory = BlurKernelEstimator()
    dir = 'C:\\Users\\zhao\\Desktop\\WeChatPortrait.png'
    # print(dir)
    pic = factory.openPic(dir)
    pic = factory.tailor(pic)
    vis.Visdom(env=factory.__class__.__name__).image(pic, win='tailored', opts=dict(title='Tailored'))
