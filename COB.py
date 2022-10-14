#imports needed libraries
import os
import sympy as sy
import matplotlib.pyplot as plt
import numpy
import unittest

def changeOfBasis(b1,b2,x1):
    D = numpy.empty((len(b1),len(b1)))
    C = numpy.empty((len(b1),len(b1)))
    T = 1
    for i in range(0,len(b1)):
        for j in range(0,len(b1[0])):
            C[i,j] = numpy.inner(b2[:,i],b1[:,j])
            D[i,j] = numpy.inner(b2[:,j],b2[:,i])
            c = x1
    ans = numpy.matmul(numpy.matmul(numpy.linalg.inv(D),C),c)
    return ans,T

class Test_changeOfBasis( unittest.TestCase ):
    def test_standardR2BasisRotate( self ):
        b1 = numpy.eye(2)
        b2 = numpy.array([ [0, 1], [-1, 0] ] ).T
        x1 = numpy.array( [0.5, 0.5] ).T
        x2, T = changeOfBasis( b1, b2, x1 )
        v1 = b1 @ x1
        v2 = b2 @ x2
        self.assertTrue( numpy.allclose( v1, v2 ) )