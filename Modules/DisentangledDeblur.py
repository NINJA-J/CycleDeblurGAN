from torch import nn
from Modules.Loss import *
import numpy as np
import torch
from Modules.Common import *
import os

class DisentangleDeblur(nn.Module):
    def __init__(self, sImage=512, rContent=64, rBlur=16, group='Disentangle'):
        super(DisentangleDeblur, self).__init__()
        self.sImage = sImage
        self.sSharpRep = rContent
        self.sBlurRep = rBlur

        self.group = group

        sCnnLayer = 3
        sResLayer = 4 - 1
        bCnnLayer = 4
        mImgSize = sImage / pow(2, sCnnLayer)

        # sharp Content Encoder
        self.sEContent = ContentEncoder(cnnLayers=sCnnLayer, resLayers=sResLayer).cuda()
        # blur  Content Encoder
        self.bEContent = ContentEncoder(cnnLayers=sCnnLayer, resLayers=sResLayer).cuda()
        # Combined Encoder Layer
        self.eContent = Resnet(layers=1, ch=self.sEContent.oChannel, lResidual=True).cuda()

        # Blur Fully Connected Network
        self.bEBlur = BlurEncoder(sImage=sImage, cnnLayer=bCnnLayer, oSize=mImgSize).cuda()

        # blur  Generator
        self.bGImage = ImageGenerator(iCh=self.sEContent.oChannel + self.bEBlur.oChannel, cnnLayers=sCnnLayer, resLayers=sResLayer).cuda()
        # sharp Generator
        self.sGImage = ImageGenerator(iCh=self.sEContent.oChannel + self.bEBlur.oChannel, cnnLayers=sCnnLayer, resLayers=sResLayer).cuda()

        # Sharp Image Discriminator
        self.sImageD = ImageDiscriminator(scales=5, layers=5, iSize=sImage).cuda()

        self.optmTotal = torch.optim.Adam([
            dict(params=self.sEContent.parameters()),
            dict(params=self.bEContent.parameters()),
            dict(params=self.bEBlur.parameters()),
            dict(params=self.sGImage.parameters()),
            dict(params=self.bGImage.parameters()),
            dict(params=self.sImageD.parameters())
        ], lr=0.00005)

        self.data = None

        # Cycle/Content Loss
        self.cLossFn = ContentLoss(nn.MSELoss())
        self.cLoss = None
        # GAN Loss
        self.gLossFn = GANLoss()
        self.gLoss = None
        self.dLoss = None
        # Perceptual Loss
        self.pLossFn = PerceptualLoss(nn.MSELoss())
        self.pLoss = None
        # KL Divergence Loss
        self.klLossFn = KLDivergenceLoss()
        self.klLoss = None
        # Total Loss
        self.tLoss = None

        self.iSharp, self.iBlur = None, None
        self.mSharp, self.mBlur = None, None
        self.oSharp, self.oBlur = None, None
        self.mBlurAttr = None

    def forward_(self, iSharp, iBlur):

        # print("%s %s" % (iSharp.size(), iBlur.size()))

        sContent = self.eContent(self.sEContent(iSharp))
        bContent = self.eContent(self.bEContent(iBlur))
        bBlur    = self.bEBlur(iBlur)

        # print("%s %s %s" % (sContent.size(), bContent.size(), bBlur.size()))

        oSharp = self.sGImage(torch.cat([bContent, bBlur], 1))
        oBlur  = self.bGImage(torch.cat([sContent, bBlur], 1))
        # print("%s %s" % (oSharp.size(), oBlur.size()))

        return oSharp, oBlur, bBlur

    def forward(self, input):
        self.data = input

        self.iSharp = input['sharp1'].cuda()
        self.iBlur = input['blur2'].cuda()

        # sharp1,blur1  -> iSharp -> mBlur ->  oSharp
        # blur2,sharp2  -> iBlur  -> mSharp -> oBlur

        self.mSharp, self.mBlur, self.mBlurAttr = self.forward_(self.iSharp, self.iBlur)
        self.oSharp, self.oBlur, _              = self.forward_(self.mSharp, self.mBlur)

    def dBackward(self):
        iSEval = self.sImageD(self.iSharp)
        mSEval = self.sImageD(self.mSharp.detach())
        oSEval = self.sImageD(self.oSharp.detach())
        self.dLoss = \
            self.gLossFn(iSEval, 1) + \
            self.gLossFn(mSEval, 0) + \
            self.gLossFn(oSEval, 0)
        # self.loss['dLoss'] = dLoss.item()

        self.dLoss.backward()

    def backward(self):
        for i in range(4):
            self.optmTotal.zero_grad()
            self.dBackward()
            self.optmTotal.step()

        self.optmTotal.zero_grad()

        # Generative Loss
        self.gLoss = \
            self.gLossFn(self.sImageD(self.mSharp), 1) + \
            self.gLossFn(self.sImageD(self.oSharp), 1)

        # Perceptual Loss
        self.pLoss = \
            self.pLossFn(self.iBlur, self.mSharp) + \
            self.pLossFn(self.iSharp, self.mBlur)

        # Content/Cycle Loss
        self.cLoss = \
            self.cLossFn(self.iSharp, self.oSharp) + \
            self.cLossFn(self.iBlur, self.oBlur)

        # KL Divergence Loss
        self.klLoss = self.klLossFn(self.mBlurAttr, torch.randn(self.mBlurAttr.size()).cuda())

        # Total Loss
        self.tLoss = \
            self.gLoss + \
            self.cLoss * 10 + \
            self.pLoss * 0.1 + \
            self.klLoss * 0.01

        self.tLoss.backward()
        self.optmTotal.step()

    def save(self, group=None):
        if group is None:
            group = self.group

        folderPath = os.path.join("CheckPoints", group)
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
            print("Created Path %s" % folderPath)

        for name, value in self._modules.items():
            filePath = "CheckPoints/%s/%s.pth" % (group, name)
            saveInfo = "Saving %s.pth" % name
            if not os.path.exists(filePath):
                os.mknod(filePath)
                saveInfo += ", Created"
            torch.save(value.cpu().state_dict(), filePath)
            value.cuda()
            saveInfo += ", Saved"
            print(saveInfo)

    def load(self, group='Disentangle'):
        for name, value in self._modules.items():
            loadInfo = "Load %s.pth" % name
            filePath = "CheckPoints/%s/%s.pth" % (group, name)
            if os.path.exists(filePath):
                try:
                    value.load_state_dict(torch.load(filePath))
                    loadInfo += ', Loaded'
                except Exception as e:
                    loadInfo += ', Failed'
                value.cuda()
            else:
                loadInfo += ', Not Found'
            print(loadInfo)

    def getImages(self):
        images = [
            self.iSharp.detach()[0].cpu().numpy(),
            self.mBlur.detach()[0].cpu().numpy(),
            self.oSharp.detach()[0].cpu().numpy(),
            self.iBlur.detach()[0].cpu().numpy(),
            self.mSharp.detach()[0].cpu().numpy(),
            self.oBlur.detach()[0].cpu().numpy()
        ]
        for i in range(6):
            images[i] = images[i] * 0.5 + 0.5
            images[i] *= 255

        return images, 3

    def getLoss(self):
        return dict(
            gLoss=self.gLoss.item() if self.gLoss is not None else 0,
            dLoss=self.dLoss.item() if self.dLoss is not None else 0,
            pLoss=self.pLoss.item() if self.pLoss is not None else 0,
            cLoss=self.cLoss.item() if self.cLoss is not None else 0,
            klLoss=self.klLoss.item() if self.klLoss is not None else 0,
            tLoss=self.tLoss.item() if self.tLoss is not None else 0
        )

class ContentEncoder(nn.Module):
    def __init__(self, cnnLayers=3, resLayers=3):
        super(ContentEncoder, self).__init__()

        CNN = Cnn(layers=cnnLayers, sCh=64, oCh=None, eCh=None)
        self.model = nn.Sequential(*[
            CNN,
            Resnet(layers=resLayers, ch=CNN.oChannel, lResidual=True, uDropout=True)
        ])


        self.oChannel = CNN.oChannel

    def forward(self, input):
        return self.model(input)

class BlurEncoder(nn.Module):
    def __init__(self, sImage=512, cnnLayer=4, oSize=None):
        super(BlurEncoder, self).__init__()

        self.oSize = oSize
        if oSize is None:
            self.oSize = sImage / pow(2, cnnLayer-1)

        CNN = Cnn(layers=cnnLayer, iCh=3, sCh=64, eCh=None, oCh=None)

        self.model = nn.Sequential(*[
            CNN,
            Flattern(),
            nn.Linear(int(pow(sImage / pow(2, cnnLayer), 2)), int(pow(self.oSize, 2))),
            Flattern(deFlattern=True)
        ])
        self.oChannel = CNN.oChannel

    def forward(self, input):
        return self.model(input)

class ImageGenerator(nn.Module):
    def __init__(self, iCh=256, cnnLayers=3, resLayers=4):
        super(ImageGenerator, self).__init__()

        self.model = nn.Sequential(*[
            Resnet(layers=resLayers, ch=iCh),
            Cnn(layers=cnnLayers, iCh=iCh, sCh=None, eCh=64, oCh=3, transpose=True, isLast=True)
        ])

    def forward(self, input):
        return self.model(input)

class ImageDiscriminator(nn.Module):
    def __init__(self, scales, layers, iSize):
        super(ImageDiscriminator, self).__init__()

        multiScale = MixMultiScale(layers=layers, scales=scales, iCh=3, sCh=32, iSize=iSize, mCh=256)
        mix = [
            nn.Conv2d(multiScale.oChannel, 1, kernel_size=3, padding=1),
            nn.MaxPool2d(int(multiScale.oSize)),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*([multiScale] + mix))

    def forward(self, input):
        return self.model(input)


if __name__=="__main__":
    test = DisentangleDeblur(sImage=128)

    for name, value in test._modules.items():
        if issubclass(value.__class__, nn.Module):
            print("%s = %s" % (name, value))



