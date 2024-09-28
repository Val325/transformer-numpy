import math
import numpy as np
import matplotlib.pylab as plt

class PositionalEncoding():
    def __init__(self, length_sentence_max, dimension):
        self.Sequential_Length_Max = length_sentence_max
        self.Dimension_Positional_Encoding = dimension
        self.Pos_Encode = np.zeros((self.Sequential_Length_Max, self.Dimension_Positional_Encoding)) 

    def forward(self):
        for pos in range(self.Sequential_Length_Max):
            for i in range(self.Dimension_Positional_Encoding//2):
                theta = pos / (10000 ** ((2*i)/self.Dimension_Positional_Encoding))
                self.Pos_Encode[pos, 2*i ] = math.sin(theta)
                self.Pos_Encode[pos, 2*i + 1] = math.cos(theta)
        
        return self.Pos_Encode 

    def add(self, embedding):
        PosEncode.forward()
        return self.Pos_Encode + embedding 

    def test(self):
        im = plt.imshow(PosEncode.forward(), aspect='auto')
        plt.title("Positional Encoding")
        plt.xlabel("Encoding Dimension")
        plt.ylabel("Position Index")
        plt.colorbar(im)
        plt.show()

PosEncode = PositionalEncoding(128, 512)
PosEncode.test()
