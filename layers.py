from Array import Array

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        ret = Array.zeros_like(x)   # return array
        self.mask = ret             # indexes to be saved
        for i in range(ret.m):
            row = [max(0,item[0]) for item in zip(x.rows[i])]
            ret[i] = row
            idx = [item[0]>0 for item in zip(x.rows[i])]
            self.mask[i] = idx

        return ret

    def backward(self, dout):
        dx = Array.zeros_like(dout)
        

        return dx

class Affine:

class Softmax:
