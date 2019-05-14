import visdom
import time
from torch.utils import data
import torch
# import numpy as np
import traceback

class TrainHandler(object):
    def __init__(self, dataSet, model, name='train'):
        self.dataSet = dataSet
        self.dataLoader = None
        self.model = model
        self.name = name

        self.vis = visdom.Visdom(env=name)

        self.constInfo = """
        {{EpochProgress}}<br>
        {{IterProgress}}<br>
        Avg Iter Time: {{AvgIterTime}}<br>
        Avg Epoch Time: {{AvgEpochTime}}<br>
        Total Time: {{TotalTrainTime}}<br>
        Model {{SaveProgress}}
        """
        self.constInfoArgs = dict(
            EpochProgress="Epoch Initializing...",
            IterProgress="Iter Initializing...",
            AvgIterTime="#",
            AvgEpochTime="#",
            TotalTrainTime="Total Waiting For Train to Start",
            SaveProgress="---"
        )
        self.dispInfo(self.constInfoArgs)


        self.loadInfo = ""
        self.appendInfo("Train Handler Initializing ...")

        self.loss = dict(x=[])

        loss = self.model.getLoss()
        for key in loss.keys():
            self.loss[key] = []

        self.iterTimeCount = 0
        self.epochTimeCount = 0
        self.avgIterTime = 0
        self.avgEpochTime = 0

        self.appendInfo("Train Handler Initialized")

    def appendInfo(self, info):
        self.loadInfo += ("<br>" if len(self.loadInfo) > 0 else "") + info
        self.vis.text(self.loadInfo, win='console', opts=dict(title='Console Like'))
        print(info)

    def addIterTime(self, time, len=20):
        self.iterTimeCount = (self.iterTimeCount + 1) if self.iterTimeCount < len else len
        avg = self.avgIterTime = (self.avgIterTime * (self.iterTimeCount - 1) + time) / self.iterTimeCount
        self.dispInfo(dict(
            AvgIterTime="%ds : %dms" % (avg, (avg * 1000) % 1000)
        ))

    def addEpochTime(self, time, len=20):
        self.epochTimeCount = (self.epochTimeCount + 1) if self.epochTimeCount < len else len
        avg = self.avgEpochTime = (self.avgEpochTime * (self.epochTimeCount - 1) + time) / self.epochTimeCount
        self.dispInfo(dict(
            AvgEpochTime="%dm : %ds : %dms" % (avg / 60, avg % 60, (avg * 1000) % 1000)
        ))

    def dispInfo(self, info):
        constInfo = self.constInfo
        for key in info.keys():
            if key in self.constInfoArgs.keys():
                self.constInfoArgs[key] = info[key]
        for key in self.constInfoArgs.keys():
            constInfo = constInfo.replace("{{%s}}" % key, str(self.constInfoArgs[key]))
        self.vis.text(constInfo, win='Info', opts=dict(title='Info'))

    def dispImages(self):
        images, row = self.model.getImages()
        import numpy as np
        print(np.max(images[4]), " - ", np.min(images[4]))

        self.vis.images(images, nrow=3, win='Images', opts=dict(title='Image'))

    def dispLoss(self, progress):
        loss = self.model.getLoss()
        self.loss['x'].append(progress)
        for key in loss.keys():
            self.loss[key].append(loss[key])
            self.vis.line(
                X=self.loss['x'], Y=self.loss[key],
                opts=dict(
                    title=key,
                    xlabel='iter',
                    ylabel='loss'
                ), win=key
            )

    def train(self, epochs=20, batchSize=20, nWorkers=5, load=False):
        self.dataLoader = torch.utils.data.DataLoader(self.dataSet, batch_size=batchSize, shuffle=False, num_workers=nWorkers)
        self.appendInfo("Data Loader Created")

        if load:
            self.model.load()
            self.dispInfo(dict(
                SaveProgress="Loaded"
            ))

        dataSize = len(self.dataSet)

        trainStartTime = time.time()
        self.appendInfo("Start Training!")

        try:
            for epoch in range(1, epochs + 1):
                epochStartTime = time.time()
                self.dispInfo(dict(
                    EpochProgress="Epoch: %d / %d - %.2f%%" % (epoch, epochs, epoch / epochs * 100)
                ))

                dataCount = 0
                for data in self.dataLoader:
                    iterStartTime = time.time()
                    trainTime = time.time() - trainStartTime

                    dataInc = len(data['sharp1'])
                    dataCount += dataInc

                    self.dispInfo(dict(
                        IterProgress="Iter : %d / %d - %.3f%%" % (dataCount, dataSize, dataCount / dataSize * 100),
                        TotalTrainTime="%dh : %dm : %ds" % (trainTime / 3600, (trainTime % 3600) / 60, trainTime % 60)
                    ))

                    # Main Functions
                    self.model.forward(data)
                    self.model.backward()
                    # Main Func Ended

                    iterTime = time.time()-iterStartTime
                    self.addIterTime(iterTime/dataInc)

                    if dataCount % (batchSize * 5) == 0:
                        self.dispImages()
                        self.dispLoss(epoch + dataCount / dataSize)

                    if dataCount % (batchSize * 50) == 0:
                        self.model.save()
                        self.dispInfo(dict(
                            SaveProgress="Saved After Epoch.%d, Iter.%d" % (epoch, dataCount)
                        ))

                epochTime = time.time() - epochStartTime
                self.addEpochTime(epochTime)

                self.model.save()
        except Exception as e:
            self.appendInfo("!!!--- Error Occurred ---!!!")
            traceback.print_exc()
