import numpy
import Quadrature
import unittest
import Basis
import Mesh

def assembleGramMatrix(domain,degree,solution_basis):
  M = numpy.zeros(shape = (degree + 1, degree + 1))
  for A in range(0,degree + 1):
    NA = lambda x: solution_basis(x,degree,A,domain)
    for B in range(0, degree + 1):
      NB = lambda x: solution_basis(x,degree,B,domain)
      integrand = lambda x: NA(x) * NB(x)
      M[A,B] = Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * degree + 1) / 2)), domain)
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

# class Test_assembleGramMatrix( unittest.TestCase ):
    # def test_quadratic_legendre( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalLegendreBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0], [0.0, 0.0, 0.2] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_cubic_legendre( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = Basis.evalLegendreBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [ 0.0, 0.0, 0.0, 1.0/7.0] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_linear_bernstein( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 1, solution_basis = Basis.evalBernsteinBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_quadratic_bernstein( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalBernsteinBasis1D )
    #     print(test_gram_matrix)
    #     gold_gram_matrix = numpy.array( [ [0.2, 0.1, 1.0/30.0], [0.1, 2.0/15.0, 0.1], [1.0/30.0, 0.1, 0.2] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    # def test_cubic_bernstein( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = Basis.evalBernsteinBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1.0/7.0, 1.0/14.0, 1.0/35.0, 1.0/140.0], [1.0/14.0, 3.0/35.0, 9.0/140.0, 1.0/35.0], [1.0/35.0, 9.0/140.0, 3.0/35.0, 1.0/14.0], [ 1.0/140.0, 1.0/35.0, 1.0/14.0, 1.0/7.0] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_linear_lagrange( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 1, solution_basis = Basis.evalLagrangeBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_quadratic_lagrange( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalLagrangeBasis1D )
    #     gold_gram_matrix = numpy.array( [ [2.0/15.0, 1.0/15.0, -1.0/30.0], [1.0/15.0, 8.0/15.0, 1.0/15.0], [-1.0/30.0, 1.0/15.0, 2.0/15.0] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    # def test_cubic_lagrange( self ):
    #     test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = Basis.evalLagrangeBasis1D )
    #     gold_gram_matrix = numpy.array( [ [8.0/105.0, 33.0/560.0, -3.0/140.0, 19.0/1680.0], [33.0/560.0, 27.0/70.0, -27.0/560.0, -3.0/140.0], [-3.0/140.0, -27.0/560.0, 27.0/70.0, 33/560.0], [ 19.0/1680.0, -3.0/140.0, 33.0/560.0, 8.0/105.0] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )