from PretrainedThreeLayerNet import PretrainedThreeLayerNet
from functions import *
from fileloader import *
root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))

X, T = load_pgm()
W1, W2, W3, b1, b2, b3 = load_param(root + '/param.txt')
network = PretrainedThreeLayerNet(W1, W2, W3, b1, b2, b3)
num_data = len(X)

loops = 10
ep0 = 0.1
accuracy_cnt = [0]*loops
for i in range(num_data):
    x = X[i]
    t = T[i]
    for j in range(loops):
        # dLdx: derivative of L
        g = network.gradient(x,t)
        # small pertubation epsilon
        ep = Array(g.m, g.n, False)
        ep.rows = [math.copysign(ep0,item[0]) for item in zip(g.rows)]

        # input pgm with small pertubation ep
        xhat = X[i] + ep
        # limit range within 0~1
        x = in_range(xhat, 0, 1)

        # predict with pertubation
        y = network.predict(x, T[i])
        # get max
        p = argmax(y)
        # compare with teacher signal
        if p == T[i]:
            accuracy_cnt[j] += 1
        print('Current Acc Count: ' + str(accuracy_cnt[j]))

for i in range(loops):
    print("Loop: %i" %i)
    print("Accuracy: " + str(float(accuracy_cnt[i]) / num_data))
