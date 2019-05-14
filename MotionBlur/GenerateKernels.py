from MotionBlur.BlurKernel import BlurKernel
from PIL import Image as Img

import numpy as np
import os
import re
import visdom

# ytr ytrbjut  /home/ytr   172.18.3.124


def loadLastKernels(kernels, lens):
    pat = re.compile(r'psf_(\d+)_(\d+)_(\d+).bin')
    for root, _, files in os.walk('..\\Data\\Kernels'):
        for file in files:
            mat = pat.match(file)
            if mat:
                i, j, k = \
                    int(mat.groups()[0]),\
                    int(mat.groups()[1]),\
                    int(mat.groups()[2])
                if "psf_%d_%d_%d.bin" % (i, j, k) != file:
                    continue

                kernel = BlurKernel(64)
                kernel.loadPSF(os.path.join(root, file))
                if kernel is None:
                    print("Kernel Turns None when loading %s" % file)
                else:
                    kernels[i, j, int(lens[i, j])] = kernel
                    lens[i, j] += 1
    # for i in range()
    #     # break


if __name__ == '__main__':
    vis = visdom.Visdom(env='GeneratorKernels')
    step = 4
    canvas = 64
    kernels = np.array([BlurKernel(64)]*2000).reshape((10, 10, 20))
    lens = np.zeros((10, 10))
    lens.astype(np.uint)

    overview = np.zeros((640, 640))

    loadLastKernels(kernels, lens)

    # print(kernels.shape)
    # print(type(kernels))
    # print(type(kernels[2]))
    # print(type(kernels[2, 3]))

    kCount = np.sum(lens)
    iters = 0
    while kCount < 2000 and iters < 50000:
        iters += 1
        kernel = BlurKernel(64)
        kernel.genMotion()
        x, y = kernel.size
        x, y = int(x/4), int(y/4)
        if not (0 <= x < 10 and 0 <= y < 10):
            continue

        kernel.genPSF()

        if lens[x, y] < 20:
            kernels[x, y, int(lens[x, y])] = kernel
            lens[x, y] += 1
            kCount += 1
        elif 20 <= lens[x, y] < 50:
            kernels[x, y, np.random.choice(20)] = kernel
            lens[x, y] += 1

        if iters % 100 == 0:
            str = "iters: %d, kCount: %d" % (iters, kCount)
            print(str)
            vis.heatmap(lens, win='GenKernelHeatMap', opts=dict(
                title='Gen HeatMap',
                xmin=0,
                xmax=20
            ))
            vis.text(str, win='GenKernelInfo', opts=dict(
                title='GenKernelInfo'
            ))

    print("Generate Finished")

    for i in range(10):
        for j in range(10):
            # print("len[%d, %d] = %d" % (i, j, lens[i, j]))
            for k in range(min(20, int(lens[i, j]))):
                if k == 0:
                    p = kernels[i, j, k].PSF.copy()
                    p /= np.max(p)
                    overview[i * 64: i * 64 + 64, j * 64: j * 64 + 64] = p

                fName = "psf_%d_%d_%d.bin" % (i, j, k)
                try:
                    kernels[i, j, k].savePSF("..\\Data\\Kernels\\%s" % fName)
                except Exception as e:
                    print("Error when saving file %s, Info:" % fName)
                    print(e)


    vis.image(overview/np.max(overview), win="overview", opts=dict(
        title="overview"
    ))
    overview = Img.fromarray(overview).convert('L')
    overview.save("overview.png")

    print("Save Finished")

