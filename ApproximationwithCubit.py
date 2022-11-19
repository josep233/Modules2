import numpy
import Quadrature
import unittest
import Basis
import COB
import Mesh
import scipy
import matplotlib.pyplot as plt
import BEXT
import Uspline
import argparse
import sympy

## MAIN FUNCTION
def main( target_fun, spline_space ):
    filename = "temp_uspline"
    Uspline.make_uspline_mesh( spline_space, filename )
    uspline_bext = BEXT.readBEXT( filename + ".json" )
    sol = computeSolution( target_fun, uspline_bext )
    return sol
#=============================================================================================================================================
## SECONDARY FUNCTIONS
def computeSolution( target_fun, uspline_bext ):
    M = assembleGramMatrix( uspline_bext )
    F = assembleForceVector( target_fun, uspline_bext )
    d = numpy.linalg.solve( M, F )
    return d
#=============================================================================================================================================
def assembleGramMatrix(uspline_bext):
    num_elems = BEXT.getNumElems( uspline_bext )
    num_nodes = BEXT.getNumNodes( uspline_bext )
    M = numpy.zeros(shape = (num_nodes, num_nodes))
    for elem_idx in range(0,num_elems):
        elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
        p = BEXT.getElementDegree( uspline_bext, elem_id )
        omegac = BEXT.getElementDomain( uspline_bext, elem_id )
        elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
        C = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
        for a in range(0,p+1):
            A = elem_nodes[a]
            NA = lambda x: Basis.evalSplineBasis1D(x,C,a,[-1,1])
            for b in range(0,p+1):
                B = elem_nodes[b]
                NB = lambda x: Basis.evalSplineBasis1D(x,C,b,[-1,1])
                integrand = lambda x: NA(x) * NB(x)
                M[A,B] += Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
    return M
#=============================================================================================================================================
def assembleForceVector(target_fun,uspline_bext):
  num_elems = BEXT.getNumElems( uspline_bext )
  num_nodes = BEXT.getNumNodes( uspline_bext )
  F = numpy.zeros(num_nodes)
  for elem_idx in range(0,num_elems):
    elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
    p = BEXT.getElementDegree( uspline_bext, elem_id )
    omegac = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
    C = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
    for a in range(0,p+1):
      A = elem_nodes[a]
      NA = lambda x: Basis.evalSplineBasis1D(x,C,a,[-1,1])
      integrand = lambda x: NA(x) * target_fun(COB.affineMapping(x,[-1,1],omegac))
      F[A] += Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
  return F
#=============================================================================================================================================
## UTILITY FUNCTIONS
def evaluateSolutionAt( x, coeff, uspline_bext ):
    elem_id = BEXT.getElementIdContainingPoint( uspline_bext, x )
    elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
    elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
    elem_extraction_operator = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
    sol = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n]
        sol += coeff[curr_node] * Basis.evalSplineBasis1D(x,elem_extraction_operator,n,elem_domain)        
    return sol
#=============================================================================================================================================
def computeElementFitError( target_fun, coeff, uspline_bext, elem_id ):
    domain = BEXT.getDomain( uspline_bext )
    elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
    num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) + 1 )
    abs_err_fun = lambda x : abs( target_fun( COB.affineMapping(x, [-1,1], elem_domain) ) - evaluateSolutionAt( COB.affineMapping(x, [-1,1], elem_domain), coeff, uspline_bext ) )
    abs_error = Quadrature.evaluateGaussLegendreQuadrature(abs_err_fun, num_qp, elem_domain)    
    return abs_error
#=============================================================================================================================================
def computeFitError( target_fun, coeff, uspline_bext ):
    num_elems = BEXT.getNumElems( uspline_bext )
    abs_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
        abs_error += computeElementFitError( target_fun, coeff, uspline_bext, elem_id )
    domain = BEXT.getDomain( uspline_bext )
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( target_fun(x) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error
#=============================================================================================================================================
## CLI ARGUMENT PARSING
def prepareCommandInputs( target_fun_str, domain, degree, continuity ):
    spline_space = { "domain": domain, "degree": degree, "continuity": continuity }
    target_fun = sympy.parsing.sympy_parser.parse_expr( target_fun_str )
    target_fun = sympy.lambdify( sympy.symbols( "x", real = True ), target_fun )
    return target_fun, spline_space
#=============================================================================================================================================
def parseCommandLineArguments( ):
    parser = argparse.ArgumentParser()
    parser.add_argument( "–function", "-f",   nargs = 1,   type = str,   required = True )
    parser.add_argument( "–domain", "-d",     nargs = 2,   type = float, required = True )
    parser.add_argument( "–degree", "-p",     nargs = '+', type = int,   required = True )
    parser.add_argument( "–continuity", "-c", nargs = '+', type = int,   required = True )
    args = parser.parse_args( )
    return args.function[0], args.domain, args.degree, args.continuity
#=============================================================================================================================================
## TEST CALLING FROM PYTHON
class Test_python( unittest.TestCase ):
    def test_run( self ):
        target_fun_str = "sin(pi*x)"
        domain = [ 0, 1 ]
        degree = [ 2, 2 ]
        continuity = [ -1, 1, -1 ]
        target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree, continuity )
        sol = main( target_fun, spline_space )
#=============================================================================================================================================
## EXAMPLE USAGE FROM CLI
if __name__ == "__main__":
    target_fun_str, domain, degree, continuity = parseCommandLineArguments( )
    target_fun, spline_space = prepareCommandInputs( target_fun_str, domain, degree, continuity )
#=============================================================================================================================================
class Test_assembleGramMatrix( unittest.TestCase ):
    def test_two_element_linear_bspline( self ):
        target_fun = lambda x: x**0
        spline_space = { "domain": [0, 2], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
        Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
        test_gram_matrix = assembleGramMatrix( uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/3.0, 1.0/6.0, 0.0 ],
                                          [ 1.0/6.0, 2.0/3.0, 1.0/6.0 ],
                                          [ 0.0, 1.0/6.0, 1.0/3.0 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_two_element_quadratic_bspline( self ):
        target_fun = lambda x: x**0
        spline_space = { "domain": [0, 2], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
        Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
        test_gram_matrix = assembleGramMatrix( uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/5.0, 7.0/60.0, 1.0/60.0, 0.0 ],
                                          [ 7.0/60.0, 1.0/3.0, 1.0/5.0, 1.0/60.0],
                                          [ 1.0/60.0, 1.0/5.0, 1.0/3.0, 7.0/60.0 ],
                                          [ 0.0, 1.0/60.0, 7.0/60.0, 1.0/5.0] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    def test_two_element_cubic_bspline( self ):
        spline_space = { "domain": [0, 2], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
        Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
        uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
        test_gram_matrix = assembleGramMatrix( uspline_bext = uspline_bext )
        gold_gram_matrix = numpy.array( [ [ 1.0/7.0, 7.0/80.0, 1.0/56.0, 1.0/560.0, 0.0 ],
                                          [ 7.0/80.0, 31.0/140.0, 39.0/280.0, 1.0/20.0, 1.0/560.0 ],
                                          [ 1.0/56.0, 39.0/280.0, 13.0/70.0, 39.0/280.0, 1.0/56.0 ],
                                          [ 1.0/560.0, 1.0/20.0, 39.0/280.0, 31.0/140.0, 7.0/80.0 ],
                                          [ 0.0, 1.0/560.0, 1.0/56.0, 7.0/80.0, 1.0/7.0 ] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
#=============================================================================================================================================
# class Test_assembleForceVector( unittest.TestCase ):
#     def test_const_force_fun_two_element_linear_bspline( self ):
#         target_fun = lambda x: numpy.pi
#         spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ numpy.pi / 2.0, numpy.pi, numpy.pi / 2.0 ] )
#         print(test_force_vector)
#         print(gold_force_vector)
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_linear_force_fun_two_element_linear_bspline( self ):
#         target_fun = lambda x: x
#         spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ -1.0/3.0, 0.0, 1.0/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_quadratic_force_fun_two_element_linear_bspline( self ):
#         target_fun = lambda x: x**2
#         spline_space = { "domain": [-1, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ 1.0/4.0, 1.0/6.0, 1.0/4.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_const_force_fun_two_element_quadratic_bspline( self ):
#         target_fun = lambda x: numpy.pi
#         spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ numpy.pi/3.0, 2.0*numpy.pi/3.0, 2.0*numpy.pi/3.0, numpy.pi/3.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_linear_force_fun_two_element_quadratic_bspline( self ):
#         target_fun = lambda x: x
#         spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ -1.0/4.0, -1.0/6.0, 1.0/6.0, 1.0/4.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) )

#     def test_quadratic_force_fun_two_element_quadratic_bspline( self ):
#         target_fun = lambda x: x**2
#         spline_space = { "domain": [-1, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_force_vector = assembleForceVector( target_fun = target_fun, uspline_bext = uspline_bext )
#         gold_force_vector = numpy.array( [ 2.0/10.0, 2.0/15.0, 2.0/15.0, 2.0/10.0 ] )
#         self.assertTrue( numpy.allclose( test_force_vector, gold_force_vector ) ) 
#=============================================================================================================================================
# class Test_computeSolution( unittest.TestCase ):
#     def test_cubic_polynomial_target_linear_bspline( self ):
#         # print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         spline_space = { "domain": [0, 1], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         gold_sol_coeff = numpy.array( [ 9.0/160.0, 7.0/240.0, -23.0/480.0 ] )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e0 )

#     def test_cubic_polynomial_target_quadratic_bspline( self ):
#         # print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         spline_space = { "domain": [0, 1], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         gold_sol_coeff = numpy.array( [ 1.0/120.0, 9.0/80.0, -1.0/16.0, -1.0/120.0 ] )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-1 )

#     def test_cubic_polynomial_target_cubic_bspline( self ):
#         # print( "POLY TEST" )
#         target_fun = lambda x: x**3 - (8/5)*x**2 + (3/5)*x
#         spline_space = { "domain": [0, 1], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         gold_sol_coeff = numpy.array( [ 0.0, 1.0/10.0, 1.0/30.0, -1.0/15.0, 0.0 ] )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         self.assertTrue( numpy.allclose( gold_sol_coeff, test_sol_coeff ) )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-12 )

#     def test_sin_target( self ):
#         # print( "SIN TEST" )
#         target_fun = lambda x: numpy.sin( numpy.pi * x )
#         spline_space = { "domain": [0, 1], "degree": [ 3, 3 ], "continuity": [ -1, 2, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

#     def test_erfc_target( self ):
#         # print( "ERFC TEST" )
#         target_fun = lambda x: numpy.real( scipy.special.erfc( x ) )
#         spline_space = { "domain": [-1, 1], "degree": [ 3, 1, 3 ], "continuity": [ -1, 1, 1, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )

#     def test_exptx_target( self ):
#         # print( "EXPT TEST" )
#         target_fun = lambda x: float( numpy.real( float( x )**float( x ) ) )
#         spline_space = { "domain": [-1, 1], "degree": [ 5, 5, 5, 5 ], "continuity": [ -1, 4, 0, 4, -1 ] }
#         Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
#         uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
#         test_sol_coeff = computeSolution( target_fun = target_fun, uspline_bext = uspline_bext )
#         abs_err, rel_err = computeFitError( target_fun, test_sol_coeff, uspline_bext )
#         # plotCompareFunToTestSolution( target_fun, test_sol_coeff, uspline_bext )
#         self.assertAlmostEqual( first = rel_err, second = 0, delta = 1e-2 )
