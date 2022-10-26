import sympy
import unittest
import math
import numpy
import scipy
import Basis
import joblib
import Quadrature

def riemannQuadrature(fun, num_points):
    MidPoint = 0
    for i in range(0,num_points):
        points = numpy.linspace(-1.0,1.0,num_points+1)
        MidPoint += abs(points[0] - points[1]) * (fun(points[i+1]) + fun(points[i])) / 2
    return MidPoint
#=============================================================================================================================================
def getNewtonCotesQuadrature( num_points ):
    if num_points != 7 and num_points != 0:
        z = sympy.Symbol('z')
        if num_points > 1:
            P = val = sympy.ones(num_points,1)
        else:
            P = val = sympy.zeros(num_points,1)
        xj = numpy.linspace(-1,1,num_points)
        for i in range(0,len(xj)):
            for j in range(0,num_points):
                if i != j:
                    P[i,:] *= (z - xj[j]) / (xj[i] - xj[j])
                    
            val[i,:] = sympy.integrate(P[i,:],(z,-1,1))
        val = numpy.array(val)
        return xj, val
    else:
        raise Exception("num_points_MUST_BE_INTEGER_IN_[1,6]") 

def computeNewtonCotesQuadrature(fun, num_points):
    xj,val = getNewtonCotesQuadrature(num_points)
    coeff = numpy.ones((num_points,1))
    for i in range(0,len(xj)):
        coeff[i,:] = fun(xj[i])
    integral = val * coeff
    integral = sum(integral)[0]
    return integral
#=============================================================================================================================================
def evaluateGaussLegendreQuadrature(integrand, degree, domain):
    qp,W = Quadrature.computeGaussLegendreQuadrature( degree, [-1,1] )
    integral = 0
    for i in range(0,len(qp)):
        integral += integrand(qp[i]) * W[i]
    Jacobian =  (domain[1] - domain[0]) / 2
    integral *= Jacobian
    
    return integral
    
@joblib.Memory("cachedir").cache()
def legendreInts(n,domain):
    z = sympy.Symbol('z')
    M = numpy.zeros(2*n, dtype = "double")
    for i in range(0,len(M)):
        fun = Basis.evalLegendreBasis1D(z,n,i,domain)
        M[i] = sympy.integrate(fun,(z,domain[0],domain[-1]))
    return M

def computeGaussLegendreQuadrature( n,domain ):
    M = legendreInts(n,domain)
    qp = rootsLegendreBasis(n,domain)
    W = solveLinearMomentFit( M, qp, domain )
    return qp, W

def rootsLegendreBasis( degree,domain ):
    if ( degree <= 0 ):
        raise Exception( "DEGREE_MUST_BE_NATURAL_NUMBER" )
    z = sympy.symbols( 'z', real = True )
    p = Basis.evalLegendreBasis1D(z,degree,degree,domain)
    roots = sympy.roots( p, z )
    roots = list( roots.keys() )
    roots = [float(val) for val in roots]
    roots.sort()
    return roots


def assembleLinearMomentFitSystem( degree, pts, domain ):
    z = sympy.Symbol('z')
    A = numpy.zeros( shape = ( degree + 1, len( pts ) ), dtype = "double" )
    for i in range(0,degree + 1):
        for j in range(0,len(pts)):
            A[i,j] = Basis.evalLegendreBasis1D(pts[j],i,i,domain)
    return A

def solveLinearMomentFit( M, pts, domain ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, pts, domain )
    sol = scipy.optimize.lsq_linear( A, M )
    w = sol.x
    return w

def objFun( M, pts, domain ):
    degree = len( M ) - 1
    A = assembleLinearMomentFitSystem( degree, pts, domain )
    w = solveLinearMomentFit( M, pts )
    obj_val = (M - A @ w)
    obj_val = obj_val.squeeze()
    return obj_val
#=============================================================================================================================================
# class Test_legendreInts( unittest.TestCase ):
#     def test_integrate( self ):
#         n = 3
#         M_gold = numpy.zeros( 2*n, dtype = "double" )
#         M_gold[0] = 2.0
#         M = legendreInts(n,[-1,1])
#         print(M)
#         self.assertTrue( numpy.allclose( M, M_gold ) )
#     def test_integrate2( self ):
#         n = 3
#         M_gold = numpy.zeros( 2*n, dtype = "double" )
#         M_gold[0] = 1.0
#         M = legendreInts(n,[0,1])
#         print(M)
#         self.assertTrue( numpy.allclose( M, M_gold ) )
#=============================================================================================================================================
# class Test_evaluateGaussLegendreQuadrature( unittest.TestCase ):
#     def test_integrate( self ):
#         for p in range( 1, 5 ):
#             fun = lambda x : x**p
#             actual = scipy.integrate.fixed_quad(fun,-1,1)[0]
#             calculated = evaluateGaussLegendreQuadrature(integrand=fun,degree=p+1,domain=[-1,1])
#             print(actual,calculated)
#             self.assertAlmostEqual( first = calculated, second = actual, delta = 1e-12 )
#     def test_integrate2( self ):
#         for p in range( 1, 5 ):
#             fun = lambda x : x**p
#             actual = scipy.integrate.fixed_quad(fun,0,1)[0]
#             calculated = evaluateGaussLegendreQuadrature(integrand=fun,degree=p+1,domain=[0,1])
#             print(actual,calculated)
#             self.assertAlmostEqual( first = calculated, second = actual, delta = 1e-12 )
#=============================================================================================================================================
# class Test_computeRiemannQuadrature( unittest.TestCase ):
#     def test_integrate_constant_one( self ):
#         constant_one = lambda x : x**0
#         for num_points in range( 1, 100 ):
#             self.assertAlmostEqual( first = riemannQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

#     def test_integrate_linear( self ):
#         linear = lambda x : x
#         for num_points in range( 1, 100 ):
#             self.assertAlmostEqual( first = riemannQuadrature( fun = linear, num_points = num_points ), second = 0.0, delta = 1e-12 )

#     def test_integrate_quadratic( self ):
#         linear = lambda x : x**2
#         error = []
#         for num_points in range( 1, 100 ):
#             error.append( abs( (2.0 / 3.0) - riemannQuadrature( fun = linear, num_points = num_points ) ) )
#         self.assertTrue( numpy.all( numpy.diff( error ) <= 0.0 ) )

#     def test_integrate_sin( self ):
#         sin = lambda x : math.sin(x)
#         error = []
#         for num_points in range( 1, 100 ):
#             self.assertAlmostEqual( first = riemannQuadrature( fun = sin, num_points = num_points ), second = 0.0, delta = 1e-12 )

#     def test_integrate_cos( self ):
#         cos = lambda x : math.cos(x)
#         error = []
#         for num_points in range( 1, 100 ):
#             error.append( abs( (2.0) - riemannQuadrature( fun = cos, num_points = num_points ) ) )
#         self.assertTrue( numpy.all( numpy.diff( error ) <= 0.0 ) )
#=============================================================================================================================================
# class Test_computeNewtonCotesQuadrature( unittest.TestCase ):
#     def test_integrate_constant_one( self ):
#         constant_one = lambda x : 1 * x**0
#         for degree in range( 1, 6 ):
#             num_points = degree + 1
#             self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = constant_one, num_points = num_points ), second = 2.0, delta = 1e-12 )

#     def test_exact_poly_int( self ):
#         for degree in range( 1, 6 ):
#             num_points = degree + 1
#             poly_fun = lambda x : ( x + 1.0 ) ** degree
#             indef_int = lambda x : ( ( x + 1 ) ** ( degree + 1) ) / ( degree + 1 )
#             def_int = indef_int(1.0) - indef_int(-1.0)
#             self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = poly_fun, num_points = num_points ), second = def_int, delta = 1e-12 )

#     def test_integrate_sin( self ):
#         sin = lambda x : math.sin(x)
#         for num_points in range( 1, 7 ):
#             self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = sin, num_points = 1 ), second = 0.0, delta = 1e-12 )

#     def test_integrate_cos( self ):
#         cos = lambda x : math.cos(x)
#         self.assertAlmostEqual( first = computeNewtonCotesQuadrature( fun = cos, num_points = 6 ), second = 2*math.sin(1), delta = 1e-4 )
#=============================================================================================================================================
# class Test_computeGaussLegendreQuadrature( unittest.TestCase ):
    # def test_1_pt( self ):
    #     qp_gold = numpy.array( [ 0.0 ] )
    #     w_gold = numpy.array( [ 2.0 ] )
    #     [ qp, w ] = computeGaussLegendreQuadrature( 1,[-1,1] )
    #     self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
    #     self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )

    # def test_1_pt_new_domain( self ):
    #     qp_gold = numpy.array( [ 0.5 ] )
    #     w_gold = numpy.array( [ 2.0 ] )
    #     [ qp, w ] = computeGaussLegendreQuadrature( 1,[0,1] )
    #     self.assertAlmostEqual( first = qp, second = qp_gold, delta = 1e-12 )
    #     self.assertAlmostEqual( first = w, second = w_gold, delta = 1e-12 )

    # def test_2_pt( self ):
    #     qp_gold = numpy.array( [ -1.0/numpy.sqrt(3), 1.0/numpy.sqrt(3) ] )
    #     w_gold = numpy.array( [ 1.0, 1.0 ] )
    #     [ qp, w ] = computeGaussLegendreQuadrature( 2,[-1,1]  )
    #     self.assertTrue( numpy.allclose( qp, qp_gold ) )
    #     self.assertTrue( numpy.allclose( w, w_gold ) )

    # def test_3_pt( self ):
    #     qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 5.0 ),
    #                             0.0,
    #                             +1.0 * numpy.sqrt( 3.0 / 5.0 ) ] )
    #     w_gold = numpy.array( [ 5.0 / 9.0,
    #                             8.0 / 9.0,
    #                             5.0 / 9.0 ] )
    #     [ qp, w ] = computeGaussLegendreQuadrature( 3,[-1,1]  )
    #     self.assertTrue( numpy.allclose( qp, qp_gold ) )
    #     self.assertTrue( numpy.allclose( w, w_gold ) )

    # def test_4_pt( self ):
    #     qp_gold = numpy.array( [ -1.0 * numpy.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
    #                             -1.0 * numpy.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
    #                             +1.0 * numpy.sqrt( 3.0 / 7.0 - 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ),
    #                             +1.0 * numpy.sqrt( 3.0 / 7.0 + 2.0 / 7.0 * numpy.sqrt( 6.0 / 5.0 ) ) ] )
    #     w_gold = numpy.array( [ ( 18.0 - numpy.sqrt( 30.0 ) ) / 36.0,
    #                             ( 18.0 + numpy.sqrt( 30.0 ) ) / 36.0,
    #                             ( 18.0 + numpy.sqrt( 30.0 ) ) / 36.0,
    #                             ( 18.0 - numpy.sqrt( 30.0 ) ) / 36.0 ] )
    #     [ qp, w ] = computeGaussLegendreQuadrature( 4,[-1,1]  )
    #     self.assertTrue( numpy.allclose( qp, qp_gold ) )
    #     self.assertTrue( numpy.allclose( w, w_gold ) )

    # def test_5_pt( self ):
    #     qp_gold = numpy.array( [ -1.0 / 3.0 * numpy.sqrt( 5.0 + 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
    #                             -1.0 / 3.0 * numpy.sqrt( 5.0 - 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
    #                             0.0,
    #                             +1.0 / 3.0 * numpy.sqrt( 5.0 - 2.0 * numpy.sqrt( 10.0 / 7.0 ) ),
    #                             +1.0 / 3.0 * numpy.sqrt( 5.0 + 2.0 * numpy.sqrt( 10.0 / 7.0 ) ) ] )
    #     w_gold = numpy.array( [ ( 322.0 - 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
    #                             ( 322.0 + 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
    #                             128.0 / 225.0,
    #                             ( 322.0 + 13.0 * numpy.sqrt( 70.0 ) ) / 900.0,
    #                             ( 322.0 - 13.0 * numpy.sqrt( 70.0 ) ) / 900.0, ] )
    #     [ qp, w ] = computeGaussLegendreQuadrature( 5,[-1,1]  )
    #     self.assertTrue( numpy.allclose( qp, qp_gold ) )
    #     self.assertTrue( numpy.allclose( w, w_gold ) )