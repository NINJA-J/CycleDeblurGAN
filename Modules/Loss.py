from torchvision import models
from torch import nn
import torch
# from torch.autograd import Variable


class GANLoss:
    def __init__(self, loss=nn.BCELoss()):
        self.rLabelVar = None
        self.fLabelVar = None
        self.labelVar = [None, None]
        # self.Tensor = tensor
        self.loss = loss

    def toTensor(self, input, isReal):
        # isReal = int(isReal is True)
        if self.labelVar[isReal] is None or self.labelVar[isReal].numel() != input.numel():
            self.labelVar[isReal] = torch.Tensor(input.size()).fill_(isReal)
        return self.labelVar[isReal].cuda()

    def __call__(self, input, label):
        target_tensor = self.toTensor(input, label).cuda()
        return self.loss(input, target_tensor)

class ContentLoss:
    def __init__(self, lossFn):
        self.lossFn = lossFn

    def __call__(self, fImg, rImg):
        return self.lossFn(fImg, rImg)

class PerceptualLoss:
    def __init__(self, loss):
        self.lossFn = loss
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features

        self.model = nn.Sequential()
        for i, layer in enumerate(list(cnn)):
            self.model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        self.model.cuda()

    def __call__(self, fakeIm, realIm):
        f_fake = self.model.forward(fakeIm)
        f_real = self.model.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.lossFn(f_fake, f_real_no_grad)
        return loss

class KLDivergenceLoss:
    def __init__(self, useNNLoss=True, loss=None):
        self.lossFn = nn.KLDivLoss() if useNNLoss else loss

    def __call__(self, input, target):
        softmax = nn.LogSoftmax()
        return self.lossFn(softmax(input), target)
