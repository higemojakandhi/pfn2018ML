from Array import Array
from layers import *
from functions import *
from collections import OrderedDict
from fileloader import *

class PretrainedThreeLayerNet:

    def __init__(self, W1, W2, W3, b1, b2, b3):
        self.params = {}
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])

        self.lastLayer = Softmax()

    def predict(self, x, t):
        for layer in self.layers.values():
            x = layer.forward(x)

        return self.lastLayer.forward(x, t)

    def gradient(self, x, t):
        y = self.predict(x,t)

        dout = y
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        return dout
