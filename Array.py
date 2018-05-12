import operator
import sys
import math
import unittest

class ArrayError(Exception):
    pass

class Array(object):
    def __init__(self, m, n, init=True):
        if init:
            self.rows = [[0]*n for x in range(m)]
        else:
            self.rows = []
        self.m = m
        self.n = n

    ###
    # what: get access to a certain row,column
    # A[1][2]
    def __getitem__(self, idx):
        return self.rows[idx]

    ###
    # what: set a number in certain row,column
    # A[1][2]=2
    def __setitem__(self, idx, item):
        self.rows[idx] = item

    ###
    # checks if the arrays are same size
    # A==B  # true/false
    def __eq__(self, arr):
        return (arr.rows == self.rows)

    ###
    # C = A + B
    # DOES NOT modify its own array
    def __add__(self, arr):
        # error if the ranks are not equal
        if self.shape() != arr.shape():
            raise ArrayError("Trying to add Array of varying rank!")
        # create an Array instance with computed sum of each rows,columns
        ret = Array(self.m, self.n)
        for x in range(self.m):
            row = [sum(item) for item in zip(self.rows[x], arr[x])]
            ret[x] = row

        return ret

    ###
    # C = self - B
    # DOES NOT modify its own array
    def __sub__(self, arr):
        # error if the ranks are not equal
        if self.shape() != arr.shape():
            raise ArrayError("Trying to add Arrays of varying rank!")
        # create an Array instance with computed subtract of each rows,columns
        ret = Array(self.m, self.n)
        for x in range(self.m):
            row = [item[0]-item[1] for item in zip(self.rows[x], arr[x])]
            ret[x] = row

        return ret

    ###
    # C = self*B
    # DOES NOT modify its own array
    def __mul__(self, arr):
        # Error if ranks of A column & B row do not match
        arrm, arrn = arr.shape()
        if (self.n != arrm):
            raise ArrayError("arrays cannot be multipled!")

        arr_t = arr.getTranspose()
        mularr = Array(self.m, arrn)

        for x in range(self.m):
            for y in range(arr_t.m):
                mularr[x][y] = sum([item[0]*item[1] for item in zip(self.rows[x], arr_t[y])])

        return mularr

    ###
    # Private Class Method
    @classmethod
    def _makeArray(cls, rows):
        try:
            m = len(rows)
            n = len(rows[0])
            # Checks if every row has same column length
            if any([len(row) != n for row in rows[1:]]):
                raise ArrayError("inconsistent row length")
        except:
            n = len(rows)
            m = 1
            rows = [rows]
        arr = Array(m,n, init=False)
        arr.rows = rows
        return arr

    ###
    # Make Zeros Matrix
    @classmethod
    def zeros(cls, m, n):
        rows = [[0]*n for x in range(m)]
        return cls.fromList(rows)

    ###
    # Make Identty Matrix
    @classmethod
    def eye(cls, m):    # eye as in matlab
        rows = [[0]*m for x in range(m)]
        idx = 0
        for row in rows:
            row[idx] = 1
            idx += 1
        return cls.fromList(rows)

    @classmethod
    def zeros_like(cls, arr):
        return cls.zeros(arr.m, arr.n)

    ###
    # Create an array from list
    # Array.fromList([[1 2 3], [4,5,6], [7,8,9]])
    @classmethod
    def fromList(cls, listoflists):
        rows = listoflists[:]
        return cls._makeArray(rows)

    ###
    # Transpose of itself
    # CHANGE its own array
    def transpose(self):
        self.m, self.n = self.n, self.m
        self.rows = [list(item) for item in zip(*self.rows)]
        return self

    ###
    # what: transpose of an array
    #       and DOES NOT change its own array
    def getTranspose(self):
        m, n = self.n, self.m
        arr = Array(m, n)
        arr.rows =  [list(item) for item in zip(*self.rows)]

        return arr

    ###
    # returns the shape of an array (matrix)
    # A.shape   # (2,4)
    def shape(self):
        return (self.m, self.n)

    def reshape(self):
        pass

    def relu(self):
        ret = Array(self.m, self.n)
        for x in range(self.m):
            row = [max(0,item[0]) for item in zip(self.rows[x])]
            ret[x] = row
        return ret

    def softmax(self):
        ret = Array(self.m, self.n)
        sum=0
        exps = [0]*self.m
        for x in range(self.m):
            exps[x] = math.exp(self.rows[x][0])
            sum += exps[x]
        for x in range(self.m):
            ret[x] = [exps[x]/sum]
        return ret

    def argmax(self):
        if self.m==1:
            return rows.index(max(rows))
        elif self.n==1:
            rows = [list(item) for item in zip(*self.rows)]
            return rows[0].index(max(rows[0]))+1
        else:
            raise ArrayError("Not a Vector")


class ArrayTests(unittest.TestCase):

    def testAdd(self):
        m1 = Array.fromList([[1, 2, 3], [4, 5, 6]])
        m2 = Array.fromList([[7, 8, 9], [10, 11, 12]])
        m3 = m1 + m2
        self.assertTrue(m3 == Array.fromList([[8, 10, 12], [14,16,18]]))

    def testSub(self):
        m1 = Array.fromList([[1, 2, 3], [4, 5, 6]])
        m2 = Array.fromList([[7, 8, 9], [10, 11, 12]])
        m3 = m2 - m1
        self.assertTrue(m3 == Array.fromList([[6, 6, 6], [6, 6, 6]]))

    def testMul(self):
        m1 = Array.fromList([[1, 2, 3], [4, 5, 6]])
        m2 = Array.fromList([[7, 8], [10, 11], [12, 13]])
        self.assertTrue(m1 * m2 == Array.fromList([[63, 69], [150, 165]]))
        self.assertTrue(m2*m1 == Array.fromList([[39, 54, 69], [54, 75, 96], [64, 89, 114]]))

    def testTranspose(self):
        m1 = Array.fromList([[1,2,3], [4,5,6]])
        m2 = Array.fromList([[1,4], [2,5], [3,6]])
        m3 = m1.getTranspose()
        m4 = m2.getTranspose()
        self.assertTrue(m2==m3)
        self.assertTrue(m1==m4)
        # Also test getTranspose
        r3 = m3.shape()
        self.assertTrue(r3==(3,2))

    def testEye(self):

        m1 = Array.eye(3)
        m2 = Array.fromList([[1, 2, 3], [4, 5, 6], [7,8,9]])
        m3 = m2*m1
        self.assertTrue(m3 == m2)

    def testZerosLike(self):
        m1 = Array.fromList([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        m2 = Array.fromList([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        m3 = Array.zeros_like(m1)
        self.assertTrue(m3 == m2)

    def testCompare(self):
        m1 = Array.fromList([-5,-4,-3,-2,-1,0,1,2,3,4,5])
        m2 = m1 <= 0

    def testReLu(self):
        m1 = Array.fromList([-5,-4,-3,-2,-1,0,1,2,3,4,5])
        m2 = m1.relu()
        m3 = Array.fromList([0,0,0,0,0,0,1,2,3,4,5])
        self.assertTrue(m2==m3)

        m4 = m1.getTranspose()
        m5 = m4.relu()
        m6 = Array.fromList([[0],[0],[0],[0],[0],[0],[1],[2],[3],[4],[5]])
        self.assertTrue(m5==m6)

    def testSoftmax(self):
        m1 = Array.fromList([0, 1, 2, 3])
        m2 = m1.softmax()
        sums = sum(m2.rows[0])
        self.assertTrue(sums==1)

if __name__ == "__main__":
    unittest.main()
