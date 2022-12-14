import numpy
import Quadrature
import unittest
import Basis
import COB
import Mesh
import scipy
import matplotlib.pyplot as plt

def assembleGramMatrix(domain,degree,solution_basis):
  M = numpy.zeros(shape = (degree + 1, degree + 1))
  for A in range(0,degree + 1):
    NA = lambda x: solution_basis(x,degree,A,[-1,1])
    for B in range(0, degree + 1):
      NB = lambda x: solution_basis(x,degree,B,[-1,1])
      integrand = lambda x: NA(x) * NB(x)
      M[A,B] = Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * degree + 1) / 2)), domain)
  return M
#=============================================================================================================================================
def assembleForceVector(target_fun,degree,solution_basis,domain):
  F = numpy.zeros(degree + 1)
  for A in range(0,degree + 1):
    NA = lambda x: solution_basis(x,degree,A,[-1,1])
    integrand = lambda x: NA(x) * target_fun(COB.affineMapping(x,[-1,1],domain))
    F[A] = Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * degree + 1) / 2)), domain)
  return F
#=============================================================================================================================================
def computeGalerkinApproximation(target_fun,domain,degree,solution_basis):
  M = assembleGramMatrix(domain,degree,solution_basis)
  F = assembleForceVector(target_fun,degree,solution_basis,domain)
  d = numpy.linalg.inv(M) @ F
  return d
#=============================================================================================================================================
def evaluateSolutionAt( x, domain, coeff, solution_basis ):
    degree = len( coeff ) - 1
    y = 0.0
    for n in range( 0, len( coeff ) ):
        y += coeff[n] * solution_basis(x,degree,n,domain )
    return y
#=============================================================================================================================================
def computeFitError( gold_coeff, test_coeff, domain, solution_basis ):
    err_fun = lambda x: abs( evaluateSolutionAt( x, domain, gold_coeff, solution_basis ) - evaluateSolutionAt( x, domain, test_coeff, solution_basis ) )
    abs_err, _ = scipy.integrate.quad( err_fun, domain[0], domain[1], epsrel = 1e-12, limit = 1000 )
    return abs_err
#=============================================================================================================================================
def plotCompareGoldTestSolution( gold_coeff, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], domain, gold_coeff, solution_basis )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()
#=============================================================================================================================================
def plotCompareFunToTestSolution( fun, test_coeff, domain, solution_basis ):
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = fun( x[i] )
        yt[i] = evaluateSolutionAt( x[i], domain, test_coeff, solution_basis )
    plt.plot( x, y )
    plt.plot( x, yt )
    plt.show()
#=============================================================================================================================================
class Test_assembleGramMatrix( unittest.TestCase ):
    def test_quadratic_legendre( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalLegendreBasis1D )
        gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0], [0.0, 0.0, 0.2] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_cubic_legendre( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = Basis.evalLegendreBasis1D )
        gold_gram_matrix = numpy.array( [ [1.0, 0.0, 0.0, 0.0], [0.0, 1.0/3.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0], [ 0.0, 0.0, 0.0, 1.0/7.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_linear_bernstein( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 1, solution_basis = Basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_bernstein( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [0.2, 0.1, 1.0/30.0], [0.1, 2.0/15.0, 0.1], [1.0/30.0, 0.1, 0.2] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_bernstein( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = Basis.evalBernsteinBasis1D )
        gold_gram_matrix = numpy.array( [ [1.0/7.0, 1.0/14.0, 1.0/35.0, 1.0/140.0], [1.0/14.0, 3.0/35.0, 9.0/140.0, 1.0/35.0], [1.0/35.0, 9.0/140.0, 3.0/35.0, 1.0/14.0], [ 1.0/140.0, 1.0/35.0, 1.0/14.0, 1.0/7.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_linear_lagrange( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 1, solution_basis = Basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1.0/3.0, 1.0/6.0], [1.0/6.0, 1.0/3.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_quadratic_lagrange( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 2, solution_basis = Basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [2.0/15.0, 1.0/15.0, -1.0/30.0], [1.0/15.0, 8.0/15.0, 1.0/15.0], [-1.0/30.0, 1.0/15.0, 2.0/15.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    def test_cubic_lagrange( self ):
        test_gram_matrix = assembleGramMatrix( domain = [0, 1], degree = 3, solution_basis = Basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [8.0/105.0, 33.0/560.0, -3.0/140.0, 19.0/1680.0], [33.0/560.0, 27.0/70.0, -27.0/560.0, -3.0/140.0], [-3.0/140.0, -27.0/560.0, 27.0/70.0, 33/560.0], [ 19.0/1680.0, -3.0/140.0, 33.0/560.0, 8.0/105.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
#=============================================================================================================================================
class Test_assembleForceVector( unittest.TestCase ):
    def test_legendre_const_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: numpy.pi, domain = [0, 1], degree = 1, solution_basis = Basis.evalLegendreBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi, 0.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_legendre_linear_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solution_basis = Basis.evalLegendreBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi + 1.0, 1.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_legendre_quadratic_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 1, solution_basis = Basis.evalLegendreBasis1D )
        gold_force_vector = numpy.array( [ 1.0/3.0, 1.0/6.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 2, solution_basis = Basis.evalLegendreBasis1D )
        gold_force_vector = numpy.array( [ 1.0/3.0, 1.0/6.0, 1.0/30.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_lagrange_const_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: numpy.pi, domain = [0, 1], degree = 1, solution_basis = Basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi / 2.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_linear_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solution_basis = Basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi/2.0 + 1.0/3.0, numpy.pi/2.0 + 2.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_lagrange_quadratic_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 1, solution_basis = Basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ 1.0/12.0, 1.0/4.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 2, solution_basis = Basis.evalLagrangeBasis1D )
        gold_force_vector = numpy.array( [ -1.0/60.0, 1.0/5.0, 3.0/20.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

    def test_bernstein_const_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: numpy.pi, domain = [0, 1], degree = 1, solution_basis = Basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi / 2.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_linear_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: 2*x + numpy.pi, domain = [0, 1], degree = 1, solution_basis = Basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ numpy.pi/2.0 + 1.0/3.0, numpy.pi/2.0 + 2.0/3.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
    
    def test_bernstein_quadratic_force_fun( self ):
        test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 1, solution_basis = Basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ 1.0/12.0, 1.0/4.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
        test_force_vector = assembleForceVector( target_fun = lambda x: x**2.0, domain = [0, 1], degree = 2, solution_basis = Basis.evalBernsteinBasis1D )
        gold_force_vector = numpy.array( [ 1.0/30.0, 1.0/10.0, 1.0/5.0 ] )
        self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )
#=============================================================================================================================================
# class Test_computeGalerkinApproximation( unittest.TestCase ):
#     def test_cubic_polynomial_target( self ):
#         print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         domain = [ 0, 1 ]
#         degree = 2
#         solution_basis = Basis.evalBernsteinBasis1D
#         test_sol_coeff = computeGalerkinApproximation(target_fun,domain,degree,solution_basis)
#         gold_sol_coeff = numpy.array( [ 1.0 / 20.0, 1.0 / 20.0, -1.0 / 20.0 ] )
#         fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
#         # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
#         plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-12 )

    # def test_sin_target( self ):
    #     # print( "SIN TEST" )
    #     target_fun = lambda x: numpy.sin( numpy.pi * x )
    #     domain = [ 0, 1 ]
    #     degree = 2
    #     solution_basis = Basis.evalBernsteinBasis1D
    #     test_sol_coeff = computeGalerkinApproximation(target_fun,domain,degree,solution_basis)
    #     gold_sol_coeff = numpy.array( [ (12*(numpy.pi**2 - 10))/(numpy.pi**3), -(6*(3*numpy.pi**2 - 40))/(numpy.pi**3), (12*(numpy.pi**2 - 10))/(numpy.pi**3)] )
    #     fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
    #     print(fit_err)
    #     # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [0, 1], solution_basis )
    #     plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
    #     self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-1 )
        
    # def test_erfc_target( self ):
    #     # print( "ERFC TEST" )
    #     target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
    #     domain = [ -2, 2 ]
    #     degree = 3
    #     solution_basis = Basis.evalBernsteinBasis1D
    #     test_sol_coeff = computeGalerkinApproximation( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
    #     gold_sol_coeff = numpy.array( [ 1.8962208131568558391841630949727, 2.6917062016799657617278998883219, -0.69170620167996576172789988832194, 0.10377918684314416081583690502732] )
    #     fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
    #     # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-2, 2], solution_basis )
    #     plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
    #     self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-0 )
    
    # def test_exptx_target( self ):
    #     # print( "EXPT TEST" )
    #     target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
    #     domain = [ -1, 1 ]
    #     degree = 5
    #     solution_basis = Basis.evalBernsteinBasis1D
    #     test_sol_coeff = computeGalerkinApproximation( target_fun = target_fun, domain = domain, degree = degree, solution_basis = solution_basis )
    #     gold_sol_coeff = ( [ -0.74841381974620419634327921170757, -3.4222814978197825394922980704166, 7.1463655364038831935841354617843, -2.9824200396151998304868767455064, 1.6115460899636204992283970407553, 0.87876479932866366847320748048494 ] )
    #     fit_err = computeFitError( gold_sol_coeff, test_sol_coeff, domain, solution_basis )
    #     # plotCompareGoldTestSolution( gold_sol_coeff, test_sol_coeff, [-1, 1], solution_basis )
    #     plotCompareFunToTestSolution( target_fun, test_sol_coeff, domain, solution_basis )
    #     self.assertAlmostEqual( first = fit_err, second = 0, delta = 1e-2 )