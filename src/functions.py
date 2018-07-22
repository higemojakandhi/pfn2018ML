from Array import Array
import unittest
import math

def softmax(x):
    ret = Array(x.m, x.n, False)
    exps = [math.exp(item[0]) for item in zip(x.rows)]
    sums = sum(exps)
    ret.rows = [item[0]/sums for item in zip(exps)]

    return ret

def argmax(x):
    return x.rows.index(max(x.rows))+1


def cross_entropy_error(y, t_idx):
    return -math.log(y[t_idx])

def in_range(x, minx, maxx):
    ret = Array(x.m, x.n, False)
    ret.rows = [min(max(minx, item[0]), maxx) for item in zip(x.rows)]
    return ret

class FunctionTests(unittest.TestCase):
    def testSoftmax(self):
        a1 = Array.fromList([1,2,3,4,5])
        a2 = softmax(a1)
        self.assertTrue(sum(a2.rows)==1)
        self.assertTrue(a1.shape() == a2.shape())

    def testCrossEntropyError(self):
        a1 = Array.fromList([1,2,3,4,5])
        t = 2
        e = cross_entropy_error(a1,t)
        print(e)

    def testArgmax(self):
        a1 = Array.fromList([1,2,3,4,5])
        ans = argmax(a1)
        self.assertTrue(ans==5)


if __name__ == "__main__":
    unittest.main()
