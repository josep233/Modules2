import numpy
import Quadrature
import unittest
import Basis
import Mesh

def assembleGramMatrix(domain,degree,solution_basis):
  M = numpy.zeros(shape = (degree + 1, degree + 1))
  for A in range(0,degree + 1):
    NA = lambda x: solution_basis(x,degree,A)
    for B in range(0, degree + 1):
      NB = lambda x: solution_basis(x,degree,B)
      integrand = lambda x: NA(x) * NB(x)
      qp,W = Quadrature.computeGaussLegendreQuadrature( numpy.ceil((2 * degree + 1) / 2) )
      for i in range(0,len(qp)):
        M[A,B] += integrand(qp[i]) * W[i]
  return M

#issues with this code: solution_basis and Quadrature functions operate over a certain domain. COB is needed to correctly integrate or approximate.

def assembleForceVector(force_function,degree,eval_basis,domain):
  F = numpy.zeros(degree + 1)
  for A in range(0,degree + 1):
    NA = lambda x: eval_basis( degree, A, x)
    integrand = lambda x: NA(x) * force_function(basis.affine_mapping(domain,[0,1],x))
    F[A] = Quadrature.quad( integrand, domain)
  return F

#same problems as above

#for all of the above functions, when using quadrature, you are integrating two functions with power p. Thus, the degree of quadrature must be p + p. 

class Test_assembleGramMatrix( unittest.TestCase ):
    def test_quadratic_legendre( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalLegendreBasis1D )
        gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0], [0.0, 0.0, 0.2] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )