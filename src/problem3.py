from PretrainedThreeLayerNet import PretrainedThreeLayerNet
from functions import *
from fileloader import *
import random
root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))

X, T = load_pgm()
W1, W2, W3, b1, b2, b3 = load_param(root + '/param.txt')
network = PretrainedThreeLayerNet(W1, W2, W3, b1, b2, b3)
num_data = len(X)

ep0 = 0.1
accuracy_cnt = 0
accuracy_cnt_base = 0
for i in range(num_data):
    # dLdx: derivative of L
    g = network.gradient(X[i], T[i])

    # small pertubation epsilon
    ep = Array(g.m, g.n, False)
    ep.rows = [math.copysign(ep0,item[0]) for item in zip(g.rows)]
    # baseline pertubation
    ep_base = Array(g.m, g.n, False)
    ep_base.rows = [math.copysign(ep0,random.choice([-1,1])) for item in zip(g.rows)]

    # input pgm with small pertubation ep
    xhat = X[i] + ep
    # limit range within 0~1
    xhat_norm = in_range(xhat, 0, 1)
    # baseline
    xhat_base = X[i] + ep_base
    xhat_base_norm = in_range(xhat_base, 0, 1)

    # predict with pertubation
    y = network.predict(xhat_norm, T[i])
    y_base = network.predict(xhat_base_norm, T[i])
    # get max
    p = argmax(y)
    p_base = argmax(y_base)
    # compare with teacher signal
    if p == T[i]:
        accuracy_cnt += 1
    if p_base == T[i]:
        accuracy_cnt_base += 1

    print("Saving %i.pgm..." % i)
    save_pgm(xhat_norm.scalarby(255), i)
    save_pgm(xhat_base_norm.scalarby(255), i, dir='/problem3_baseline')


print("Accuracy:" + str(float(accuracy_cnt) / num_data))
print("Accuracy:" + str(float(accuracy_cnt_base) / num_data))
