from PretrainedThreeLayerNet import PretrainedThreeLayerNet
from functions import *
from fileloader import *
root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))


X, T = load_pgm()
W1, W2, W3, b1, b2, b3 = load_param(root + '/param.txt')
network = PretrainedThreeLayerNet(W1, W2, W3, b1, b2, b3)
num_data = len(X)

accuracy_cnt = 0
for i in range(num_data):
    # predict if x==t
    y = network.predict(X[i], T[i])
    # get max
    p = argmax(y)
    if p == T[i]:
        accuracy_cnt += 1
    print('Current Acc Count: ' + str(accuracy_cnt))

print("Accuracy:" + str(float(accuracy_cnt) / num_data))
