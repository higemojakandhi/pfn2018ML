import operator
import sys
import math
import unittest


# 1列のListで全て持っておいて．
# rows & columnsの変数を持っておく
# 形が一致すれば同じ要素数の箇所を足す・引く
# dotプロダクトは

class ArrayError(Exception):
    pass

class Array(object):
    def __init__(self, m, n, init=True):
        if init:
            self.rows = [0]*m*n
        self.m = m
        self.n = n
        self.len = m*n

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
    # Print function
    def __str__(self):
        rows = [[0]*self.n for x in range(self.m)]
        for i in range(self.m):
            rows[i] = self.rows[self.n*i : self.n*(i+1)]
        s='\n'.join([' '.join(map(str, row)) for row in rows])
        return s+'\n'

    ###
    # C = A + B
    # DOES NOT modify its own array
    def __add__(self, arr):
        # error if the ranks are not equal
        if self.shape() != arr.shape():
            raise ArrayError("Trying to add Array of varying rank!")
        # create an Array instance with computed sum of each rows,columns
        ret = Array(self.m, self.n)
        ret.rows = [sum(item) for item in zip(self.rows, arr.rows)]

        return ret

    ###
    # C = A - B
    # DOES NOT modify its own array
    def __sub__(self, arr):
        # error if the ranks are not equal
        if self.shape() != arr.shape():
            raise ArrayError("Trying to subtract Array of varying rank!")
        # create an Array instance with computed sum of each rows,columns
        ret = Array(self.m, self.n)
        ret.rows = [item[0]-item[1] for item in zip(self.rows, arr.rows)]

        return ret

    ###
    # C = self*B
    # DOES NOT modify its own array
    # TODO Assumed Matrix*Vector, need to implement Mat*Mat
    # Matrix needs to be transposed
    def __mul__(self, arr):
        # Error if ranks of A column & B row do not match
        arrm, arrn = arr.shape()
        if (self.n != arrm):
            raise ArrayError("arrays cannot be multipled!")

        ret = Array(self.m, arrn)
        for i in range(self.m):
            ret.rows[i] = sum(item[0]*item[1] for item in zip(self.rows[self.n*i : self.n*(i+1)], arr.rows))
        return ret

    def __iadd__(self, arr):
        temp = self + arr
        self = temp
        return self

    ###
    # returns the shape of an array (matrix)
    # A.shape   # (2,4)
    def shape(self):
        return (self.m, self.n)

    ###
    # Transpose of itself
    # and DOES NOT change its own array
    def transpose(self):
        ret = Array(self.n, self.m)
        for i in range(self.m):
            for j in range(self.n):
                ret.rows[self.m*j+i] = self.rows[self.n*i + j]
        return ret

    def scalarby(self, scalar):
        ret = Array(self.m, self.n, False)
        ret.rows = [scalar*item[0] for item in zip(self.rows)]
        return ret


    @classmethod
    def fromList(cls, rows):
        try:
            # Assume rows = [[rand, rand], [rand, rand]]
            m = len(rows)
            n = len(rows[0])
            # Checks if every row has same column length
            if any([len(row) != n for row in rows[1:]]):
                raise ArrayError("inconsistent row length")
            rows = [r for row in rows for r in row]
        except:
            # Assume rows = [rand, rand]
            m = len(rows)
            n = 1
        arr = Array(m,n, init=False)
        arr.rows = rows
        return arr

    @classmethod
    def zeros_like(cls, arr):
        return Array(arr.m, arr.n)


class ArrayTests(unittest.TestCase):

    def testEqual(self):
        a1 = Array.fromList([[1,2,3],[4,5,6]])
        a2 = [1,2,3,4,5,6]
        a3 = Array.fromList(a2) # We Assume Column Vector
        a4 = Array.fromList([a2]) # Row Vector
        self.assertTrue(a1.rows == a2)
        self.assertTrue(a1.m == 2)
        self.assertTrue(a1.n == 3)
        self.assertTrue(a3.m == 6)
        self.assertTrue(a3.n == 1)
        self.assertTrue(a4.m == 1)
        self.assertTrue(a4.n == 6)

    def testAdd(self):
        a1 = Array.fromList([1,2,3])
        a2 = Array.fromList([4,5,6])
        a3 = Array.fromList([5,7,9])
        a4 = Array.fromList([[1,2,3],[4,5,6]])
        a5 = Array.fromList([[7,8,9],[10,11,12]])
        a6 = Array.fromList([[8,10,12],[14,16,18]])
        self.assertTrue(a3 == (a1+a2))
        self.assertTrue(a6 == (a4+a5))
        print(a4)
        print(a5)
        print(a6)

    def testSub(self):
        a1 = Array.fromList([4,5,6])
        a2 = Array.fromList([1,2,3])
        a3 = Array.fromList([3,3,3])
        self.assertTrue(a3 == (a1-a2))

    def testMul(self):
        a1 = Array.fromList([[1,2,3],[4,5,6]])
        a2 = Array.fromList([1,2,3])
        a3 = Array.fromList([14,32])
        self.assertTrue(a3 == a1*a2)

    def testTranspose(self):
        a1 = Array.fromList([[1,2,3],[4,5,6],[7,8,9]])
        a2 = a1.transpose()
        a3 = Array.fromList([[1,4,7],[2,5,8],[3,6,9]])
        self.assertTrue(a3 == a2)

if __name__ == "__main__":
    unittest.main()
