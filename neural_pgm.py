from Array import Array
import os

###
# How Data is Aligned
# X
#  1 2 3 4 ... 153 154
#  *
#  *
#  *
#
# T
#  1 2 3 4 ... 153 154
#  *
#  *
#  *
def load_pgm():
    files = os.listdir('./pgm/')
    num_file = len(files)

    # size of PGM is 32
    size = 32
    # Input Data to be stored
    X = Array(num_file, size**2)
    for i in range(num_file):
        # Open one of files in the pgm directory
        print('Opening File: pgm/' + str(i+1)+'.pgm')
        f = open('pgm/'+str(i+1)+'.pgm','r')
        # List of data in the text
        data = f.read().split("\n")

        # initialize with Zeros
        rows = [0]*size**2
        idx = 0

        for line in data:
            row = line.split(" ")
            if (len(row)==size):
                temp = [item[0]/255 for item in zip(list(map(float, row)))]
                rows[size*idx:(idx+1)*size] = [item[0]/255 for item in zip(list(map(float, row)))]
                idx += 1

        X[i] = rows

    # Teaching Data to be stored
    # T = [0]*num_file
    f_t = open('labels.txt')
    data = f_t.read().split("\n")
    T = list(map(float,data))
    print('printint T... ')
    print(T)

    f.close()
    return X, T

def load_param():
    # N=InputLayerDim, H=MiddleLayerDim, C=OutputLayerDim
    N = 1024
    H = 256
    C = 23
    # Open File
    f = open('param.txt', 'r')
    # Split into each line
    data = f.read().split("\n")
    # index
    idx = 0

    row_w1 = [[0]*N for x in range(H)]
    row_w2 = [[0]*H for x in range(H)]
    row_w3 = [[0]*H for x in range(C)]
    idx_w1 = idx_w2 = idx_w3 = idx_b1 = idx_b2 = idx_b3 = 0
    rowlist = [256, 1, 256, 1, 23, 1]
    for line_str in data:
        line = line_str.split(" ")
        if line==['']:
            break
        line = list(map(float,line))
        # W1 = H*N      # 0~H-1
        # b1 = H        # H
        # W2 = H*H      # H+1~2H
        # b2 = H        # 2H+1
        # W3 = C*H      # 2H+3 ~ 2H+C+3
        # b3 = C        # 2H+C+3
        if (idx<H):
            row_w1[idx_w1] = line
            idx_w1+=1
        elif (idx==H):
            row_b1 = line
        elif (idx>=H+1 and idx<2*H+1):
            row_w2[idx_w2] = line
            idx_w2+=1
        elif (idx==2*H+1):
            row_b2 = line
        elif (idx>=2*H+2 and idx<2*H+C+2):
            row_w3[idx_w3] = line
            idx_w3+=1
        elif (idx==2*H+C+2):
            row_b3 = line
        idx+=1
    W1 = Array.fromList(row_w1)
    b1 = Array.fromList(row_b1)
    W2 = Array.fromList(row_w2)
    b2 = Array.fromList(row_b2)
    W3 = Array.fromList(row_w3)
    b3 = Array.fromList(row_b3)

    return W1, W2, W3, b1, b2, b3

def predict(x):
    X = Array.fromList(x)
    X.transpose()
    W1, W2, W3, b1, b2, b3 = load_param()
    a1 = W1*X + b1.transpose()
    h1 = a1.relu()
    a2 = W2*h1 + b2.transpose()
    h2 = a2.relu()
    a3 = W3*h2 + b3.transpose()
    print(a3.rows)
    y = a3.softmax()

    return y

# MAIN
# Load Parameters from Files
x, t = load_pgm()
# Size of Data
m, n = x.shape()
accuracy_cnt = 0
for i in range(m):
    print('Trial: ' + str(i))
    # Data -> Neural Network -> Prediction Y
    y = predict(x[i])
    # print(y.rows)
    # Get Index of Highest Possibility Num
    p = y.argmax()
    # if Prediction matches with
    print('Prediction: ' + str(p))
    print('Teacher: ' + str(t[i]))
    if p == t[i]:
        accuracy_cnt += 1
    print('Current Acc Count: ' + str(accuracy_cnt))

print("Accuracy:" + str(float(accuracy_cnt) / m))

# def load_weights(num_nodes):
#     num_layer = len(num_nodes)
#     f = open('param.txt', 'r')
#     data = f.read().split("\n") # Split into each line
#     idx = 0     # index
#
#     rowlist=[]
#     for i in range(num_layer-1):
#         rowlist.append(num_nodes[i+1])
#         rowlist.append(1)
#     print(rowlist)
#
#     for line_str in data:
#         line = line_str.split(" ")
#         line = list(map(float,line))
#
#             rowsize     = rowlist[i]
#         for i in range(len(rowlist)):
#             startidx    = sum(rowlist[0:i])
#             endidx      = sum(rowlist[0:i+1])
#             if(idx>=startidx and idx<endidx):
#
#         idx+=1
#         if idx==sum(rowlist):
#             break
