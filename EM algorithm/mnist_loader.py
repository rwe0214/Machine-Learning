import numpy as np
from functools import reduce
import math

class load_MNIST():
    def __init__(self, image, label, bins=0):
        self.images, self.images_infos = load_MNIST.load_idx_file(image)
        self.labels, self.labels_infos = load_MNIST.load_idx_file(label)
        if bins != 0:
            self.bins = self.images >> int(math.log(256/bins,2))
        else:
            self.bins = self.images
    
    @staticmethod
    def load_idx_file(filename):
        with open(filename, mode='rb') as f:
            magic_number = f.read(4)
            dim = magic_number[-1]
            infos = [int.from_bytes(f.read(4), byteorder='big') for _ in range(dim)]
            datas = f.read(reduce(lambda x,y: x*y, infos))  
            datas = np.frombuffer(datas, dtype="uint8")  
            datas = np.reshape(datas,(infos[0],reduce(lambda x,y: x*y, (infos[1:]+[1]))))
        return datas, infos