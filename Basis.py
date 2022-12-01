import sympy
import unittest
import math
import numpy
import joblib
import COB

def evalMonomialBasis1D(degree,variate):
    x = sympy.Symbol('x') 
    func = x**degree 
    return func.subs(x,variate) 
#=============================================================================================================================================
def evalLegendreBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [-1,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    z = sympy.Symbol('z')
    legendrefunc = symLegendreBasis1D(variate,degree,basis_idx,space_domain)
    return legendrefunc.subs(z,variate)
@joblib.Memory("cachedir").cache()
def symLegendreBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    legendrefunc = 1/((2**basis_idx)*math.factorial(basis_idx)) * sympy.diff((z**2 - 1)**basis_idx,z,basis_idx)
    return legendrefunc
#=============================================================================================================================================

def evalBernsteinBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [0,1]
    z = sympy.Symbol('z')
    B = symBernsteinBasis1D(variate,degree,basis_idx,space_domain)
    variate = COB.affineMapping(variate,space_domain,param_domain)
    ans = B.subs(z,variate) 
    return ans  

def evalBernsteinBasis1DVector(variate,degree,space_domain):
    param_domain = [0,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    basis_val_vector = numpy.zeros(((degree + 1),1))
    for i in range(0,degree + 1):
        basis_val_vector[i] = evalBernsteinBasis1D( variate, degree, i, param_domain)
    return basis_val_vector

@joblib.Memory("cachedir").cache()
def symBernsteinBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    B = sympy.functions.combinatorial.factorials.binomial(degree, basis_idx) * (z**basis_idx) * (1 - z)**(degree - basis_idx)
    return B
#=============================================================================================================================================
def evalSplineBasis1D(variate,C,basis_idx,domain):
    degree = C.shape[0] - 1
    N = evalBernsteinBasis1DVector(variate,degree,domain)
    basis_val = numpy.matmul(C, N)[basis_idx]
    return basis_val
#=============================================================================================================================================
def evalLagrangeBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [-1,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    z = sympy.Symbol('z')
    P = 1
    xj = numpy.linspace(-1,1,degree+1)
    P = symsLagrangeBasis1D(variate,degree,basis_idx,space_domain)
    ans = P.subs(z,variate)
    return ans
@joblib.Memory("cachedir").cache()
def symsLagrangeBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    P = 1
    xj = numpy.linspace(-1,1,degree+1)
    for i in range(0,degree+1):
        if basis_idx != i:
            P *= (z - xj[i]) / (xj[basis_idx] - xj[i])
    return P
#=============================================================================================================================================
def evalBernsteinBasisDeriv(degree,basis_idx,deriv,domain,variate):
    if deriv >= 1:
        jacobian = ( domain[-1] - domain[0] ) / ( 1 - 0 )
        z = sympy.Symbol('z')
        basis_func = symBernsteinBasis1D(variate,degree,basis_idx,domain)
        deriv_param = sympy.diff(basis_func,z,deriv)
        # variate = COB.affineMapping(variate,[0,1],domain)
        deriv_dom = deriv_param.subs(z,variate) / jacobian
    else:
        deriv_dom = evalBernsteinBasis1D(variate,degree,basis_idx,domain)
    return deriv_dom
#=============================================================================================================================================
def evalBernsteinBasisDeriv1DVector( degree, deriv, domain, variate ):
    basis_deriv_vector = numpy.zeros( shape = ( degree + 1 ) )
    for basis_idx in range( 0, degree + 1 ):
        basis_deriv_vector[basis_idx] = evalBernsteinBasisDeriv( degree, basis_idx, deriv, domain, variate )
    return basis_deriv_vector
#=============================================================================================================================================
def evalSplineBasisDeriv1D( extraction_operator, basis_idx, deriv, domain, variate ):
    degree = extraction_operator.shape[0] - 1
    N = evalBernsteinBasisDeriv1DVector( degree, deriv, domain, variate )
    basis_deriv = numpy.matmul( extraction_operator, N )[basis_idx]
    return basis_deriv      
#=============================================================================================================================================  
class Test_evalSplineBasisDeriv1D( unittest.TestCase ):
       def test_C0_linear_0th_deriv_at_nodes( self ):
              C = numpy.eye( 2 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 1.0 )

       def test_C0_linear_1st_deriv_at_nodes( self ):
              C = numpy.eye( 2 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

       def test_C1_quadratic_0th_deriv_at_nodes( self ):
              C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.25 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.625 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 0.5 ), second = 0.125 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ 0, 1 ], variate = 1.0 ), second = 0.5 )

       def test_C1_quadratic_1st_deriv_at_nodes( self ):
              C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = -2.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.0 ), second = +0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = -1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 0.5 ), second = +0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = -1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

       def test_C1_quadratic_2nd_deriv_at_nodes( self ):
              C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +2.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = -3.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.0 ), second = +1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +2.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = -3.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 0.5 ), second = +1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +2.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = -3.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ 0, 1 ], variate = 1.0 ), second = +1.0 )

       def test_biunit_C0_linear_0th_deriv_at_nodes( self ):
              C = numpy.eye( 2 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 1.0 )

       def test_biunit_C0_linear_1st_deriv_at_nodes( self ):
              C = numpy.eye( 2 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

       def test_biunit_C1_quadratic_0th_deriv_at_nodes( self ):
              C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = -1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.25 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.625 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +0.0 ), second = 0.125 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 0, domain = [ -1, 1 ], variate = +1.0 ), second = 0.5 )

       def test_biunit_C1_quadratic_1st_deriv_at_nodes( self ):
              C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = -1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +1.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = -1.0 ), second = +0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = -0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.0 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = -0.5 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 1, domain = [ -1, 1 ], variate = +1.0 ), second = +0.5 )

       def test_biunit_C1_quadratic_2nd_deriv_at_nodes( self ):
              C = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.5 ], [ 0.0, 0.0, 0.5 ] ] )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.50 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = -0.75 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = -1.0 ), second = +0.25 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.50 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = -0.75 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +0.0 ), second = +0.25 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 0, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.50 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 1, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = -0.75 )
              self.assertAlmostEqual( first = evalSplineBasisDeriv1D( extraction_operator = C, basis_idx = 2, deriv = 2, domain = [ -1, 1 ], variate = +1.0 ), second = +0.25 )
#=============================================================================================================================================
class Test_evalBernsteinBasisDeriv( unittest.TestCase ):
       def test_constant_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

       def test_constant_1st_deriv_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

       def test_constant_2nd_deriv_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 0, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )

       def test_linear_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.0, delta = 1e-12 )

       def test_linear_at_gauss_pts( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 0, domain = [0, 1], variate =  0.5 ), second = 0.5, delta = 1e-12 )

       def test_quadratic_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 1.00, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.50, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 0.00, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.0 ), second = 0.00, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 0.5 ), second = 0.25, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = 1.0 ), second = 1.00, delta = 1e-12 )

       def test_quadratic_at_gauss_pts( self ):
              x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
              x = [ COB.affineMapping( xi, [-1, 1], [0, 1] ) for xi in x ]
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.62200846792814620, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.04465819873852045, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.33333333333333333, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.33333333333333333, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[0] ), second = 0.04465819873852045, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 0, domain = [0, 1], variate = x[1] ), second = 0.62200846792814620, delta = 1e-12 )

       def test_linear_1st_deriv_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +1.0, delta = 1e-12 )

       def test_linear_1st_deriv_at_gauss_pts( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second = +1.0, delta = 1e-12 )

       def test_linear_2nd_deriv_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.0 ), second = 0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 1.0 ), second = 0, delta = 1e-12 )

       def test_linear_2nd_deriv_at_gauss_pts( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 0, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 1, basis_idx = 1, deriv = 2, domain = [0, 1], variate = 0.5 ), second = 0, delta = 1e-12 )

       def test_quadratic_1st_deriv_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

       def test_quadratic_1st_deriv_at_gauss_pts( self ):
              x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
              x = [ COB.affineMapping( xi, [-1, 1], [0, 1] ) for xi in x ]
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[0] ), second = -1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = x[1] ), second = -1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[0] ), second = +2.0 / math.sqrt(3), delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = x[1] ), second = -2.0 / math.sqrt(3), delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[0] ), second = +1.0 - 1/( math.sqrt(3) ), delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = x[1] ), second = +1.0 + 1/( math.sqrt(3) ), delta = 1e-12 )

       def test_quadratic_2nd_deriv_at_nodes( self ):
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.0 ), second = -2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 0.5 ), second = -1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 1, domain = [0, 1], variate = 1.0 ), second =  0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.0 ), second = +2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 1, domain = [0, 1], variate = 1.0 ), second = -2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.0 ), second =  0.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 0.5 ), second =  1.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 1, domain = [0, 1], variate = 1.0 ), second = +2.0, delta = 1e-12 )

       def test_quadratic_2nd_deriv_at_gauss_pts( self ):
              x = [ -1.0 / math.sqrt(3.0) , 1.0 / math.sqrt(3.0) ]
              x = [ COB.affineMapping( xi, [-1, 1], [0, 1] ) for xi in x ]
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 0, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[0] ), second = -4.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 1, deriv = 2, domain = [0, 1], variate = x[1] ), second = -4.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[0] ), second = +2.0, delta = 1e-12 )
              self.assertAlmostEqual( first = evalBernsteinBasisDeriv( degree = 2, basis_idx = 2, deriv = 2, domain = [0, 1], variate = x[1] ), second = +2.0, delta = 1e-12 )
#=============================================================================================================================================
class Test_evaluateMonomialBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        self.assertAlmostEqual( first = evalMonomialBasis1D( degree = 0, variate = 0 ), second = 1.0, delta = 1e-12 )
        for p in range( 1, 11 ):
            self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 0 ), second = 0.0, delta = 1e-12 )
            self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 1 ), second = 1.0, delta = 1e-12 )

    def test_basisAtMidpoint( self ):
        for p in range( 0, 11 ):
            self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 0.5 ), second = 1 / ( 2**p ), delta = 1e-12 )
# #=============================================================================================================================================
class Test_evalLegendreBasis1D( unittest.TestCase ):
    def test_basisAtBounds( self ):
        for p in range( 0, 2 ):
            if ( p % 2 == 0 ):
                self.assertAlmostEqual( first = evalLegendreBasis1D(variate=-1,degree=p,basis_idx=p,space_domain=[-1,1]), second = +1.0, delta = 1e-12 )
            else:
                self.assertAlmostEqual( first = evalLegendreBasis1D(variate=-1,degree=p,basis_idx=p,space_domain=[-1,1]), second = -1.0, delta = 1e-12 )
            self.assertAlmostEqual( first = evalLegendreBasis1D(variate=+1,degree=p,basis_idx=p,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )

    def test_constant( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D(variate=x,degree=0,basis_idx=0,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )

    def test_linear( self ):
        for x in numpy.linspace( -1, 1, 100 ):
            self.assertAlmostEqual( first = evalLegendreBasis1D(variate=x,degree=1,basis_idx=1,space_domain=[-1,1]), second = x, delta = 1e-12 )

    def test_quadratic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D(variate=-1.0 / math.sqrt(3.0),degree=2,basis_idx=2,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D(variate=+1.0 / math.sqrt(3.0),degree=2,basis_idx=2,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )

    def test_cubic_at_roots( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D(variate=-math.sqrt( 3 / 5 ),degree=3,basis_idx=3,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D(variate=0,degree=3,basis_idx=3,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLegendreBasis1D(variate=+math.sqrt( 3 / 5 ),degree=3,basis_idx=3,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )

    def test_unit_domain( self ):
        self.assertAlmostEqual( first = evalLegendreBasis1D(variate=0,degree=1,basis_idx=1,space_domain=[0,1]), second = -1, delta = 1e-12 )
#=============================================================================================================================================
class Test_evaluateBernsteinBasis1D( unittest.TestCase ):
    def test_linearBernstein( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0, degree = 1, basis_idx = 0, space_domain = [0,1] ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0, degree = 1, basis_idx = 1, space_domain = [0,1] ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 1, degree = 1, basis_idx = 0, space_domain = [0,1] ), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 1, degree = 1, basis_idx = 1, space_domain = [0,1] ), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0.5, degree = 1, basis_idx = 0, space_domain = [0,1] ), second = 0.5, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0.5, degree = 1, basis_idx = 1, space_domain = [0,1] ), second = 0.5, delta = 1e-12 )


    def test_quadraticBernstein( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0, degree = 2, basis_idx = 0, space_domain = [0,1] ), second = 1.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0, degree = 2, basis_idx = 1, space_domain = [0,1] ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = 0, degree = 2, basis_idx = 2, space_domain = [0,1] ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate =  0.5, degree = 2, basis_idx = 0, space_domain = [0,1] ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate =  0.5, degree = 2, basis_idx = 1, space_domain = [0,1] ), second = 0.50, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate =  0.5, degree = 2, basis_idx = 2, space_domain = [0,1] ), second = 0.25, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 0, space_domain = [0,1] ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 1, space_domain = [0,1] ), second = 0.00, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D( variate = +1, degree = 2, basis_idx = 2, space_domain = [0,1] ), second = 1.00, delta = 1e-12 )
    def test_unit_domain( self ):
        self.assertAlmostEqual( first = evalBernsteinBasis1D(variate=0,degree=1,basis_idx=0,space_domain=[0,1]), second = 1, delta = 1e-12 )
        self.assertAlmostEqual( first = evalBernsteinBasis1D(variate=0,degree=1,basis_idx=1,space_domain=[0,1]), second = 0, delta = 1e-12 )
#=============================================================================================================================================
class Test_evaluateLagrangeBasis1D( unittest.TestCase ):
    def test_linearLagrange( self ):
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=-1,degree=1,basis_idx=0,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=-1,degree=1,basis_idx=1,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=+1,degree=1,basis_idx=0,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=1,degree=1,basis_idx=1,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )

    def test_quadraticLagrange( self ):
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=-1,degree=2,basis_idx=0,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=-1,degree=2,basis_idx=1,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=-1,degree=2,basis_idx=2,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=0,degree=2,basis_idx=0,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=0,degree=2,basis_idx=1,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=0,degree=2,basis_idx=2,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=1,degree=2,basis_idx=0,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=1,degree=2,basis_idx=1,space_domain=[-1,1]), second = 0.0, delta = 1e-12 )
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=1,degree=2,basis_idx=2,space_domain=[-1,1]), second = 1.0, delta = 1e-12 )
    def test_unit_domain( self ):
        self.assertAlmostEqual( first = evalLagrangeBasis1D(variate=0,degree=1,basis_idx=0,space_domain=[0,1]), second = 1, delta = 1e-12 )