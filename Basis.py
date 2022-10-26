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
@joblib.Memory("cachedir").cache()
def evalLegendreBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [-1,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    z = sympy.Symbol('z')
    legendrefunc = 1/((2**basis_idx)*math.factorial(basis_idx)) * sympy.diff((z**2 - 1)**basis_idx,z,basis_idx)
    return legendrefunc.subs(z,variate)
#=============================================================================================================================================

def evalBernsteinBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [0,1]
    z = sympy.Symbol('z')
    B = symBernsteinBasis1D(variate,degree,basis_idx,space_domain)
    variate = COB.affineMapping(variate,space_domain,param_domain)
    ans = B.subs(z,variate) 
    return ans
@joblib.Memory("cachedir").cache()
def symBernsteinBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    B = sympy.functions.combinatorial.factorials.binomial(degree, basis_idx) * (z**basis_idx) * (1 - z)**(degree - basis_idx)
    return B
#=============================================================================================================================================
@joblib.Memory("cachedir").cache()
def evalLagrangeBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [-1,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    z = sympy.Symbol('z')
    P = 1
    xj = numpy.linspace(-1,1,degree+1)
    for i in range(0,degree+1):
        if basis_idx != i:
            P *= (z - xj[i]) / (xj[basis_idx] - xj[i])
    ans = P.subs(z,variate)
    return ans

#=============================================================================================================================================
# class Test_evaluateMonomialBasis1D( unittest.TestCase ):
#     def test_basisAtBounds( self ):
#         self.assertAlmostEqual( first = evalMonomialBasis1D( degree = 0, variate = 0 ), second = 1.0, delta = 1e-12 )
#         for p in range( 1, 11 ):
#             self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 0 ), second = 0.0, delta = 1e-12 )
#             self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 1 ), second = 1.0, delta = 1e-12 )

#     def test_basisAtMidpoint( self ):
#         for p in range( 0, 11 ):
#             self.assertAlmostEqual( first = evalMonomialBasis1D( degree = p, variate = 0.5 ), second = 1 / ( 2**p ), delta = 1e-12 )
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