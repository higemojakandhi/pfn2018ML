from PretrainedThreeLayerNet import PretrainedThreeLayerNet
from functions import *
from fileloader import *
import random
root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))

X, T = load_pgm()
W1, W2, W3, b1, b2, b3 = load_param(root + '/param.txt')
network = PretrainedThreeLayerNet(W1, W2, W3, b1, b2, b3)
num_data = len(X)

eps0 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
accuracy_cnt = [0]*len(eps0)
ep_count = 0
for ep0 in eps0:
    for i in range(num_data):
        # dLdx: derivative of L
        g = network.gradient(X[i], T[i])

        # small pertubation epsilon
        ep = Array(g.m, g.n, False)
        ep.rows = [math.copysign(ep0,item[0]) for item in zip(g.rows)]

        # input pgm with small pertubation ep
        xhat = X[i] + ep
        # limit range within 0~1
        xhat_norm = in_range(xhat, 0, 1)

        # predict with pertubation
        y = network.predict(xhat_norm, T[i])
        # get max
        p = argmax(y)
        # compare with teacher signal
        if p == T[i]:
            accuracy_cnt[ep_count] += 1
    ep_count += 1

for i in range(len(eps0)):
    print("Accuracy:" + str(float(accuracy_cnt[i]) / num_data))
