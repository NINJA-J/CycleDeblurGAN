from torch import nn
import numpy as np
import torch

class Cnn(nn.Module):
    def __init__(self, layers=3, iCh=3, sCh=32, eCh=32, oCh=32, mCh=256, transpose=False, isLast=False):
        # self.model=[]
        super(Cnn, self).__init__()

        assert iCh is not None or sCh is not None, "iCh or sCh shouldn't be None"
        cList = []
        chs = layers - 1
        model = []
        if iCh is None:
            chs += 1
        if sCh is None:
            sCh = pow(2, np.around(np.log2(iCh) + 1))

        if eCh is None and oCh is None:
            chs += 1
            eCh = sCh * pow(2, chs - 1)
            oCh = eCh
        elif eCh is None:
            eCh = pow(2, np.around(np.log2(oCh) + 1))
        elif oCh is None:
            chs += 1


        self.iChannel = iCh
        self.oChannel = oCh

        chs = layers - 1

        for i in range(chs):
            cList.append(
                int(np.minimum(
                    np.minimum(
                        sCh * pow(2, i),
                        eCh * pow(2, chs - 1 - i)
                    ),
                    mCh)
                )
            )

        cList = \
            ([] if iCh is None else [iCh]) + \
            cList + \
            ([] if oCh is None else [oCh])

        # l = len(cList) - 1
        for i in range(len(cList) - 1):
            if transpose:
                model += [
                    nn.ConvTranspose2d(cList[i], cList[i + 1], kernel_size=3, stride=2, padding=1, output_padding=1)
                ]
            else:
                model += [
                    nn.Conv2d(cList[i], cList[i + 1], kernel_size=3, stride=2, padding=1)
                ]

            model += [
                nn.InstanceNorm2d(cList[i + 1]),
                nn.ReLU(True)
            ]
            # model += [
            #     nn.Conv2d(cList[i], cList[i+1], kernel_size=3, stride=2, padding=1) if transpose is False else
            #         nn.ConvTranspose2d(cList[i], cList[i+1], kernel_size=3, stride=2, padding=1, output_padding=1),
            #
            #     nn.InstanceNorm2d(cList[i+1]),
            #     nn.ReLU(True)  # if not isLast else nn.Tanh()
            # ]
            # lrfitsse
        if isLast:
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(3, 3, kernel_size=7, padding=0),
                nn.Tanh()
            ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class Encoder(nn.Module):
    def __init__(self, iSize=512, fLen=64, eSize=8, mChannel=256, sChannel=32):
        """
        (b, 3,              iSize,      .) -> 
        (b, sChannel,       iSize/2,    .) -> 
        ... -> 
        (b, 2 * eChannel,   eSize * 2,  .) -> 
        (b, fLen,           eSize,      .) ->
        (b, fLen,           1,          .) -> (b, fLen)
        
        :param iSize:   Input Size      输入图片大小
        :param fLen:    Feature Length  输出特征长度,等同于eChannel'（此时特征图片大小为eSize,）
        :param eSize:   End Size        AvgPooling时的特征图片大小
        :param mChannel:Max Channel     卷积层的最大Channel
        :param sChannel:Start Channel   第一层的输出oChannel
        """""
        super(Encoder, self).__init__()
        self.Cnn = Cnn(iSize=iSize, oSize=eSize, sCh=sChannel, eCh=fLen, mCh=mChannel)
        self.flat = nn.AvgPool2d(eSize)

    def forward(self, input):
        # output = self.model(input).squeeze()
        output1 = self.Cnn(input)
        output2 = self.flat(output1).squeeze()
        # print("Shape after Encoder.forward %s - %s - %s" % (input.shape, output1.shape, output2.shape))
        return output2

class Decoder(nn.Module):
    def __init__(self, iSize=512, fLen=64, mChannel=256, eChannel=32):
        super(Decoder, self).__init__()
        self.model = None
        self.fLen = fLen
        model = []

        sChannel = pow(2, np.around(np.log2(fLen)))
        layers = int(np.log2(iSize))
        cList = []

        for i in range(layers):
            channel = np.minimum(
                (pow(2, i) * sChannel) if i != 0 else fLen,
                pow(2, layers - i - 1) * eChannel
            )
            channel = int(np.minimum(channel, mChannel))
            cList.append(channel if i != 0 else fLen)
        cList.append(3)

        for i in range(1, len(cList)):
            # print("cList[%d]=%d as %s, cList[%d]=%d as %s" % (i-1, cList[i-1], str(type(cList[i-1])), i, cList[i], str(type(cList[i]))))
            model += [
                nn.ConvTranspose2d(cList[i-1], cList[i], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(cList[i]),
                nn.ReLU(True)
            ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        assert input.shape[1] == self.fLen, "Length of the Feature Mismatched %s - %s" % (input.shape[1], self.fLen)
        if len(input.shape) == 2:
            input = input.unsqueeze(2).unsqueeze(2)
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self, iSize=512):
        super(Discriminator, self).__init__()
        self.fLen = 16
        self.fLen2 = 4
        self.encoder = Encoder(iSize=iSize, fLen=self.fLen, mChannel=128, sChannel=8)
        self.fTrans = nn.Linear(self.fLen, self.fLen2)

    def forward(self, input):
        output = self.encoder(input).squeeze()
        output = self.fTrans(output).mean()
        return output

class ResnetBlock(nn.Module):
    def __init__(self, ch=256, uDropout=False, uBias=False):
        super(ResnetBlock, self).__init__()

        conv = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch, ch, kernel_size=3, bias=uBias),
            nn.InstanceNorm2d(ch)
        ]

        model = conv + [
            nn.ReLU(True)
        ] + ([
            nn.Dropout(0.5)
        ] if uDropout else []
        ) + conv

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = x + self.model(x)
        return out

class Resnet(nn.Module):
    def __init__(self, layers=3, ch=256, lResidual=False, uDropout=False, uBias=False):
        super(Resnet, self).__init__()

        model = []
        if layers < 1:
            laysers = 1
        for i in range(layers):
            model += [
                ResnetBlock(ch=ch, uDropout=uDropout, uBias=uBias)
            ]
        self.model = nn.Sequential(*model)
        self.lResidual = lResidual

    def forward(self, input):
        output = self.model(input)
        if self.lResidual:
            output = input + output
        return output

class Reshape(nn.Module):
    def __init__(self, toShape):
        self.toShape = toShape

    def forward(self, input):
        return input.reshape([input.shape[0], input.shape[1]] + self.toShape)

class Flattern(nn.Module):
    def __init__(self, deFlattern=False):
        super(Flattern, self).__init__()
        self.deFlattern = deFlattern

    def forward(self, input):
        # print("Flattern Input Size: %s" % str(input.shape))
        if self.deFlattern:
            size = int(pow(input.shape[2], 0.5))
            output = input.view([input.shape[0], input.shape[1], size, size])
        else:
            output = input.view([input.shape[0], input.shape[1], -1])

        # print("Flattern Output Size: %s" % str(output.shape))
        return output

class MixMultiScale(nn.Module):
    def __init__(self, layers=5, scales=5, iCh=3, sCh=32, iSize=512, mCh=256):
        super(MixMultiScale, self).__init__()

        self.layers = layers

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.convs = []
        self.oChannel = 0
        for i in range(scales):
            cList = [iCh, sCh]
            model = []
            for j in range(layers - i - 1):
                cList.append(
                    np.minimum(
                        sCh * pow(2, j),
                        mCh
                    )
                )
            self.oChannel += cList[-1]
            for j in range(len(cList) - 1):
                model += [
                    nn.Conv2d(cList[j], cList[j+1], kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(cList[j+1]),
                    nn.ReLU(True)
                ]
            self.convs += [nn.Sequential(*model).cuda()]
        # self.oChannel = sCh * (pow(2, layers) - 1)
        self.oSize = iSize / pow(2, layers)
        print("oChannel of MultiScale is %d, oSize is %d" % (self.oChannel, self.oSize))

    def forward(self, input):
        output = []
        for i in range(self.layers):
            out = self.convs[i](input)
            output += [out]
            input = self.pool(input)
            # print("Size of %d is %s, Size of Input is %s" % (i, str(out.shape), str(input.shape)))
        output = torch.cat(output, 1)
        return output





