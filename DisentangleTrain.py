from Modules.DisentangledDeblur import DisentangleDeblur
from DataSet.ImageDataHandler import ImageDataSet
from torch.utils import data
from visdom import Visdom
from Util.TrainHandler import TrainHandler


if __name__=="__main__":
    epochs = 40

    # 15198
    # 150

    trainName = 'Disentangle'
    dataset = 'CelebA'

    if dataset == 'CelebA':
        batchSize = 30
        numWorker = 4
        sImage = 128
    elif dataset == 'Disentangle':
        batchSize = 30
        numWorker = 25
        sImage = 256
        iterSize = 5000
    else:
        batchSize = 30
        numWorker = 25
        iterSize = 5000

        sImage = 128

    train = TrainHandler(
        # dataSet=ImageDataSet(dataFolder='./Data', group=dataset, sImage=sImage, size=iterSize, rand=True, refine=4),  # Disentangle - 16 Pics
        dataSet=ImageDataSet(dataFolder='./Data', group=dataset, sImage=sImage),  # CelebA
        model=DisentangleDeblur(sImage=sImage, rContent=64, rBlur=16, group=trainName),
        name=trainName)

    train.train(epochs=epochs, batchSize=batchSize, nWorkers=numWorker, load=True)
