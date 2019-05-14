import numpy as np
import os
import ntpath
from visdom import Visdom

import threading
import queue

class Visualizer(threading.Thread):
    def __init__(self, env='Visualizer'):
        super(Visualizer, self).__init__()
        self.name = env
        self.vis = Visdom(env=env)

        self.taskList = queue.Queue()
        self.fStop = False
        self.funcList = dict(
            result=self.dispCurrentResults,
            tTrain=self.dispTrainTime,
            tIter=self.dispIterTime,
            text=self.infoText
        )

        self.plotData = dict(
            loss=dict(
                x=[],
                y=[],
            )
        )

        self.lossData = {
            'X': [],
            'loss': []
        }
        self.trainTime = {
            'X': [],
            'Time': []
        }

        self.iterTime = {
            'iter': [],
            'time': []
        }

        self.epochIterTime = [[]]
        # self.boxEpoch = 0
        # self.boxIter = 0

    # |visuals|: dictionary of images to display or save
    #   vis.task(msg=dict(
    #       func='result',
    #       data=dict(
    #           epoch=,
    #           data=,
    #           errors=
    #       )
    #   ))
    def dispCurrentResults(self, msg):
        epoch = msg['epoch']
        data = msg['data']
        errors = msg['error']

        for item in data.keys():
            # print(data[item].shape)
            self.vis.image(data[item], win=item, opts=dict(
                title=item
            ))

        self.lossData['X'].append(epoch)
        self.lossData['loss'].append(errors['loss'])
        self.vis.line(
            X=np.array(self.lossData['X']),
            Y=np.array(self.lossData['loss']),
            opts={
                'title': 'Loss',
                'xlabel': 'Epoch',
                'ylabel': 'Loss'},
            win='loss'
        )

    #   vis.task(msg=dict(
    #       func='tTrain',
    #       data=dict(
    #           epoch=,
    #           time=
    #       )
    #   ))
    def dispTrainTime(self, msg):
        epoch = msg['epoch']
        time = msg['time']

        self.trainTime['X'].append(epoch)
        self.trainTime['Time'].append(time)
        self.vis.line(
            X=np.array(self.trainTime['X']),
            Y=np.array(self.trainTime['Time']),
            opts={
                'title': 'Training Time',
                'xlabel': 'Epoch',
                'ylabel': 'Time'
            },
            win='trainTime')

    #   vis.task(msg=dict(
    #       func='tIter',
    #       data=dict(
    #           iter=,
    #           time=
    #       )
    #   ))
    # def addIterTime(self, msg):

    def dispIterTime(self, msg):
        epoch = msg['epoch']
        time = msg['time']

        if len(self.epochIterTime) <= epoch:
            self.epochIterTime.append([time])
        else:
            self.epochIterTime[epoch].append(time)

        if epoch != 0:
            self.epochIterTime[0][0] = self.epochIterTime[0][1]
            self.vis.boxplot(
                X=np.array(self.epochIterTime[:epoch]).transpose(),
                opts={
                    'title': 'Iter Time per Epoch',
                    'xlabel': 'Epoch',
                    'ylabel': 'Time'
                },
                win='iterTime')

    def task(self, msg):
        self.taskList.put(msg)

    def run(self):
        while not self.fStop:
            if not self.taskList.empty():
                msg = self.taskList.get()
                self.funcList[msg['func']](msg['data'])

    def stop(self):
        self.fStop = False

    #   vis.task(msg=dict(
    #       func='text',
    #       data=dict(
    #           text=[],
    #           title=,     # 可选
    #           win=        # 可选
    #       )
    #   ))
    def infoText(self, msg):
        text = msg['text']
        try:
            title = msg['title']
        except:
            title = "info"
        try:
            win = msg['win']
        except:
            win = title


        text_ = ""

        if win is None:
            win = title

        if isinstance(text, str):
            text_ = text
        elif isinstance(text, list):
            text_ = ""
            for s in text:
                text_ += s + "<br>"

        self.vis.text(text_, win=win, opts=dict(
            title=title,
            font='Calibri')
        )


if __name__=="__main__":
    vis = Visdom(env="boxPlotTest")

    # uneven = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5]])

    # vis.boxplot(
    #     uneven.transpose(),
    #     win="unEven"
    # )
    even = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5],
        [6],
        [7],
        [8]
        ])
    vis.boxplot(
        even,
        win="even1",
    )

    # vis.boxplot(
    #     even.transpose(),
    #     win="even2"
    # )
