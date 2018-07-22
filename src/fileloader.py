from Array import Array
import os
import os.path
root = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir))

def load_pgm():
    files = os.listdir(root+'/pgm/')
    num_file = len(files)

    # size of PGM is 32
    size = 32
    # Input Data to be stored
    X = [Array(size**2,1, False) for i in range(num_file)]
    for i in range(num_file):
        # Open one of files in the pgm directory
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

        X[i].rows = rows

    # Teaching Data to be stored
    f_t = open('labels.txt')
    data = f_t.read().split("\n")
    T = list(map(float,data))
    f.close()

    return X, T

def load_param(filename):
    # N=InputLayerDim, H=MiddleLayerDim, C=OutputLayerDim
    N = 1024
    H = 256
    C = 23
    # Open File
    f = open(filename, 'r')
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

def save_pgm(x, num_of_file, size=32, max=255, dir='/problem3/'):
    header = '\n'.join(['P2', ' '.join(map(str, [size, size])), str(max)])
    rows = [[0]*size for i in range(size)]
    for i in range(size):
        rows[i] = x.rows[size*i : size*(i+1)]
    s ='\n'.join([' '.join(map(str, map(int, row))) for row in rows])

    save_path = root + dir
    save_file = os.path.join(save_path, str(num_of_file)+'.pgm')
    f = open(save_file,'w+')

    f.write(header + '\n' + s)

    f.close()
