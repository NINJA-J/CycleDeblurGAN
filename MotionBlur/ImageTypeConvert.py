from PIL import Image
import numpy as np
import os

def convertFileType(fFolder, tFolder = None):
    if tFolder is None:
        tFolder = fFolder

    for root, _, files in os.walk(fFolder):
        i = 0
        for file in files:
            newName = "sharpImg_%d.png" % i
            Image.open(os.path.join(root, file)).convert('RGB').save(os.path.join(tFolder, newName))
            print("Convert File: %s ==> %s" % (file, newName))
            i += 1

if __name__ == '__main__':
    # convertFileType('.\\SharpImages')
    pass
