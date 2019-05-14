from DataSet.ImageDataHandler import ImageDataLoader, ImageDataSet
from Modules.Procedures import BlurKernelEstimator
from Modules.visualizer import Visualizer
from Util.util import progress

import time
import os

# GPU 0.02-0.6
# CPU 1.2-1.9

if __name__ == '__main__':
    vis = Visualizer(env='MotionDeblur')
    # print("Visualizer Loaded")
    vis.start()

    # print("Visualizer Initialized")

    epochs = 20
    batchSize = 2

    # count of Epoch
    cEpoch = 0
    # count of Iter
    cIter = 0

    model = BlurKernelEstimator(sPhoto=256)
    dataSet = ImageDataSet(sImage=256, sKernel=64, size=200)
    dataLoader = ImageDataLoader(dataSet, batchSize=batchSize, nWorkers=2)
    dataLen = len(dataSet)

    # if os.path.exists("CheckPoints\\model.pth"):
    model.load()

    sTrainTime = time.time()

    for cEpoch in range(epochs):
        print("Epoch %d" % cEpoch)
        sEpochTime = time.time()
        cIter = 0.0

        for data in dataLoader:
            sIterTestTime = time.time()
            cIter += batchSize
            if cIter > dataLen:
                cIter = dataLen
            # print("Iter %d" % cIter)

            tTrainTime = time.time()-sTrainTime
            vis.task(msg=dict(
                func='text',
                data=dict(
                    text=[
                        "Epoch %d / %d" % (cEpoch, epochs),
                        progress((cEpoch + 1) / epochs),
                        "Iter  %d / %d" % (cIter, dataLen),
                        progress(cIter / dataLen, line=2),
                        "Total Train Time: %d: %d: %d" % (tTrainTime/60, tTrainTime % 60, (tTrainTime*100) % 100)
                    ],
                    title='Progress'
                )
            ))

            # vis.infoText([
            #     "Epoch %d / %d" % (cEpoch, epochs),
            #     progress(cEpoch + 1 / epochs),
            #     "Iter  %d / %d" % (cIter, dataLen),
            #     progress(cIter / dataLen, line=3)
            # ], title='Progress')

            # count of Epoch(Float)
            cEpochF = cEpoch + cIter / dataLen

            sIterTime = time.time()
            model.forward(data)
            model.backward()
            eIterTime = time.time() - sIterTime
            vis.task(msg=dict(
                func='tIter',
                data=dict(
                    epoch=cEpoch,
                    time=eIterTime
                )
            ))
            # vis.dispIterTime(cIter, (time.time() - sIterTime) / 60)

            if cIter % 50 == 0:
                vis.task(msg=dict(
                    func='result',
                    data=dict(
                        epoch=cEpochF,
                        data=model.currData(),
                        error=model.currError()
                    )
                ))
                # vis.dispCurrentResults(cEpochF, model.currData(), model.currError())

            eIterTestTime = time.time() - sIterTestTime
            print("Iter No.%d time: %.2f / %.2f   %.2f%%" % (cIter, eIterTime, eIterTestTime, eIterTime/eIterTestTime))

        model.save()
        print("Model Saved After Epoch %d" % (cEpoch + 1))
        vis.task(msg=dict(
            func='tTrain',
            data=dict(
                epoch=cEpoch,
                time=time.time() - sEpochTime
            )
        ))
        # vis.dispTrainTime(cEpoch, (time.time() - sEpochTime) / 60)

    model.save()
    vis.stop()

