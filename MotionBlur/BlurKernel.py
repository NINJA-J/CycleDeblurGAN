import numpy as np
# from math import ceil
# import matplotlib.pyplot as plt
# import cv2
# import os
from scipy import signal
# from scipy import misc
# from MotionBlur.generate_trajectory import Motion  # ,Trajectory
from PIL import Image as Img
import itertools
from visdom import Visdom
from skimage import restoration
# import prep
from PIL.PngImagePlugin import PngImageFile




class BlurKernel(object):
    def __init__(self, canvas=64, psf=None):
        self.size = None
        self.motion = None
        if psf is not None:
            psf = np.array(psf)
            assert psf.size[0] == psf.size[1], "Kernel should be a square shape"
            self.PSF = psf
            self.canvas = psf.size[0]
        else:
            self.PSF = np.zeros((canvas, canvas))
            self.motion = np.array([])
            self.canvas = 64
            self.size = (0, 0)

        # self.psfSum = None

    def genMotion(self, iters=2000, maxLen=60, expl=None):
        expl = 0.1 * np.random.uniform(0, 1) if expl is None else expl
        # self.x = None

        tLength = 0
        bigExplCount = 0

        # 向心力
        centripetal = 0.7 * np.random.uniform(0, 1)
        # probability of big shake
        pBigShake = 0.2 * np.random.uniform(0, 1)
        # term determining, at each sample, the random component of the new direction
        gausShake = 10 * np.random.uniform(0, 1)
        initAngle = np.pi * np.random.uniform(0, 1)

        v = complex(np.sin(initAngle), np.cos(initAngle)) * (
            expl if expl > 0 else maxLen / (iters - 1))

        x = np.array([complex(0, 0)] * iters)

        for t in range(0, iters - 1):
            if np.random.uniform() < pBigShake * expl:
                nextDirection = 2 * v * (np.exp(complex(0, np.pi + (np.random.uniform() - 0.5))))
                bigExplCount += 1
            else:
                nextDirection = 0

            dv = nextDirection + expl * (
                    gausShake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]
            ) * (maxLen / (iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (maxLen / float((iters - 1)))
            x[t + 1] = x[t] + v
            tLength = tLength + abs(x[t + 1] - x[t])

        # centere the motion
        x += complex(
            self.canvas / 2 - (np.max(x.real) + np.min(x.real)) / 2,
            self.canvas / 2 - (np.max(x.imag) + np.min(x.imag)) / 2
        )
        self.motion = x
        self.size = (
            int(np.max(self.motion.real)-np.min(self.motion.real)),
            int(np.max(self.motion.imag)-np.min(self.motion.imag))
        )

        return self.motion

    def saveMotion(self, name='motion2.bin'):
        assert self.motion is not None
        self.motion.tofile(name)

    def loadMotion(self, name='motion2.bin'):
        self.motion = np.fromfile(name, dtype=complex)
        self.motionSize = (
            int(np.max(self.motion.real) - np.min(self.motion.real)),
            int(np.max(self.motion.imag) - np.min(self.motion.imag))
        )
        return self.motion

    def genPSF(self, fraction=1):
        # print("DType of Motion is %s" % str(self.motion.dtype))

        # def clamp(x, xMin=0, xMax=1):
        #     return xMin if x < xMin else x if x <= xMax else xMax
        #
        # def map(x, aMin, aMax, bMin, bMax):
        #     return bMin + (x - aMin) / (aMax - aMin) * (bMax - bMin)

        if self.motion is None:
            self.genMotion()

        PSF = np.zeros((self.canvas, self.canvas))
        iters = len(self.motion)

        for t in range(iters):

            x, y = self.motion[t].real, self.motion[t].imag
            if not (0 <= x < self.canvas - 1 and 0 <= y < self.canvas - 1):
                continue

            cords = [
                range(int(x), int(x + 2)),
                range(int(y), int(y + 2))
            ]
            #           a       x4
            #   x0 ----------------------- x1
            #   |               |          |
            #   |               |          |
            #  b|               |          |
            #   |               |          |
            #   | --------------x6-------- |
            #   |               |          |
            #   x2 ----------------------- x3
            #                   x5
            # x4 = a * x1 + (1 - a) * x0
            # x5 = a * x3 + (1 - a) * x2
            # x6 = b * x5 + (1 - b) * x4
            #    = a * b * x3 + (1 - a) * b * x2 + a * (1 - b) * x1 + (1 - a)*(1 - b) * x0
            #                                                         ====================
            for xt, yt in itertools.product(*cords):
                val = (1 - np.abs(x - xt)) * (1 - np.abs(y - yt))
                PSF[yt, xt] += val

        self.PSF = PSF / np.sum(PSF)

        return self.PSF

    def savePSF(self, pathToSave='..\\Data\\Kernels\\PSF.bin'):
        imgPathToSave = pathToSave[:-3] + 'png'
        Img.fromarray(self.PSF/np.max(self.PSF)*255).convert('L').save(imgPathToSave)

        psf = np.asarray(self.PSF)
        psf.tofile(pathToSave)

    def savePNGTest(self):
        Img.fromarray(self.PSF*255).convert('L').save("testPng.jpg")
        kernel = Img.open("testPng.jpg").convert('L')
        kernel = np.asarray(kernel)/255

        sum = 0
        for i in range(64):
            for j in range(64):
                sum += pow(np.abs(self.PSF[i, j]-kernel[i, j]), 1)

        print("The Sum of Errors is %.5f / %.5f" % (sum, np.sum(np.asarray(self.PSF))))

    def loadPSF(self, pathToLoad='..\\Data\\Kernels\\PSF.bin'):
        psf = np.fromfile(pathToLoad)
        psf.shape = (self.canvas, self.canvas)

        self.PSF = psf

    def showPSF(self):
        Img.fromarray(self.PSF/np.max(self.PSF)*255).convert('L').show()
        print('Type of PSF is %s' % str(self.PSF.dtype))

    def openImg(self, img=None):
        image = img
        if isinstance(image, str):
            image = Img.open(image).convert('RGB')
            image = np.asarray(image)
        elif isinstance(image, list):
            image = np.asarray(image)
            if len(image.shape) != 3:
                raise Exception('Not a Valid Image.')
        else:
            image = image.convert('RGB')
            image = np.asarray(image)
        # elif not isinstance(image, np.ndarray):
        #     raise Exception('Image Type Unknown.')

        # if image.shape[0] != 3:
        #     if image.shape[2] != 3:
        #         raise Exception('Current Image is not a RGB File')
        #     else:
        #         image = image.transpose((2, 0, 1))

        # print(image.shape)

        return image

    def conv(self, image, kernel, method='same'):
        for i in range(3):
            if image.ndim == 3:
                image[i] = np.array(signal.fftconvolve(image[i], kernel, method))
            else:
                kernel[i] = np.array(signal.fftconvolve(image, kernel[i], method))
        return image

    def blur(self, img=None):
        kernel = self.PSF
        image = self.openImg(img).copy()

        trans = False

        if image.shape[0] != 3:
            image = image.transpose((2, 0, 1))
            trans = True

        for i in range(3):
            image[i] = np.array(signal.fftconvolve(image[i], kernel, 'same'))

        if trans:
            return image.transpose((1, 2, 0))

    def deblur(self, img=None, iter=100):
        print("delburing img")

        useRL = True

        kernel = self.PSF
        bImage = self.openImg(img)
        sImage = bImage.copy()

        if useRL:
            for i in range(3):
                mean = np.mean(sImage[i])
                rng = np.max(sImage[i])-np.min(sImage[i])
                sImage[i] = (sImage[i] - mean) * 2 / rng
                sImage[i] = restoration.richardson_lucy(sImage[i], self.PSF)
                sImage[i] = sImage[i] * rng / 2 + mean
        else:
            bImage.astype(np.double)

            """
                                            bImage
            sImage = sImage * conv( ------------------------, kernel.T)
                                      conv(sImage, kernel)
            """

            for i in range(iter):

                print("iter %d" % i)

                tmp = self.conv(sImage, kernel)
                tmp = bImage / tmp
                tmp = self.conv(tmp, kernel.transpose())
                # print("ndim of tmp is %d, kernel is %d, kernelT is %d" % (tmp.ndim, kernel.ndim, kernel.transpose().ndim))
                # tmp = self.conv(kernel.transpose(), tmp)
                print("Sum of tmp is %.2f, max is %.2f, min is %.2f" % (np.sum(tmp), np.max(tmp), np.min(tmp)))
                sImage = sImage * tmp

                # sImage = sImage * self.conv(bImage / self.conv(sImage, kernel), kernel.transpose())
        return sImage


if __name__ == '__main__':
    vis = Visdom(env='BlurKernel')

    psf = BlurKernel(canvas=64)
    psf.loadPSF("..\\Data\\Kernels\\psf_7_3_15.bin")

    kernel = psf.PSF/np.max(psf.PSF)*255
    vis.image(kernel, win="kernel", opts=dict(title='kernel'))

    bImage = psf.blur('C:\\Users\\zhao\\Desktop\\WeChatImage.png')
    vis.image(bImage, win="bImage", opts=dict(title='bImage'))

    # sImage2 = psf.deblur(bImage, 2)
    # vis.image(sImage2, win="sImage2", opts=dict(title='sImage2'))
    #
    # sImage5 = psf.deblur(bImage, 3)
    # vis.image(sImage5, win="sImage5", opts=dict(title='sImage5'))
    #
    # sImage10 = psf.deblur(bImage, 5)
    # vis.image(sImage10, win="sImage10", opts=dict(title='sImage10'))

    sImage1000 = psf.deblur(bImage, 1000)

    vis.image(sImage1000, win="sImage1000", opts=dict(title='sImage1000'))




