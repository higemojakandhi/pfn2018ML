from Array import Array
from functions import *
import unittest

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        ret = Array.zeros_like(x)   # return array
        self.mask = ret             # indexes to be saved
        ret.rows = [max(0,item[0]) for item in zip(x.rows)]
        self.mask = [item[0]>0 for item in zip(x.rows)]

        return ret

    def backward(self, dout):
        dx = dout
        for i in range(dx.len):
            if self.mask[i]==False:
                dx[i] = 0

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        self.x = x
        return self.W*x + self.b

    def backward(self, dout):
        return self.W.transpose()*dout

class Softmax:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        return self.y

    def backward(self, dout):
        delta = Array.zeros_like(dout)
        delta[int(self.t)-1] = 1
        return dout - delta


class LayerTests(unittest.TestCase):
    def testRelu(self):
        a1 = Array.fromList([-5,-4,-3,-2,-1,0,1,2,3,4,5])
        r1 = Relu()
        a2 = r1.forward(a1)
        a3 = Array.fromList([0,0,0,0,0,0,1,2,3,4,5])
        self.assertTrue(a2 == a3)

        a4 = Array.fromList([1,2,3,4,5,6,7,8,9,10,11])
        a5 = r1.backward(a4)
        a6 = Array.fromList([0,0,0,0,0,0,7,8,9,10,11])
        self.assertTrue(a5 == a6)

if __name__ == "__main__":
    unittest.main()
