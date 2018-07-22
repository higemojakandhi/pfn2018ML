from PretrainedThreeLayerNet import PretrainedThreeLayerNet
from collections import OrderedDict
from collections import Counter
from functions import *
from fileloader import *
import os
root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))

X, T = load_pgm()
num_data = len(X)
networks = OrderedDict()
# create networks from parameters defined in the directory
num_file = len(os.listdir(root + '/extra/'))
for i in range(num_file):
    name = 'param_' + str(i)
    W1, W2, W3, b1, b2, b3 = load_param(root + '/extra/' + name + '.txt')
    networks[name] = PretrainedThreeLayerNet(W1, W2, W3, b1, b2, b3)

ep0 = 0.1
accuracy_cnt = 0
for i in range(num_data):
    # dLdx: derivative of L
    g = networks['param_0'].gradient(X[i], T[i])

    # small pertubation epsilon
    ep = Array(g.m, g.n, False)
    ep.rows = [math.copysign(ep0,item[0]) for item in zip(g.rows)]

    # input pgm with small pertubation ep
    xhat = X[i] + ep
    # limit range within 0~1
    xhat_norm = in_range(xhat, 0, 1)

    # predict with pertubation
    y = [networks['param_'+str(i)].predict(xhat_norm, T[i]) for i in range(num_file)]
    # get max
    p = [argmax(y[i]) for i in range(num_file)]
    print(p)
    c = Counter(p)
    value, count = c.most_common()[0]
    print(value)
    # compare with teacher signal
    if value == T[i]:
        accuracy_cnt += 1
    print('Current Acc Count: ' + str(accuracy_cnt))

print("Accuracy:" + str(float(accuracy_cnt) / num_data))
