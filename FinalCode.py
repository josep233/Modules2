import numpy
import scipy
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
import Quadrature
import unittest
import Basis
import COB
import Mesh
import BEXT
import Uspline
import argparse
import sympy

## MAIN CODE
def computeSolution(problem, uspline_bext):
    K = assembleStiffnessMatrix( problem, uspline_bext )
    F = assembleForceVector(problem,uspline_bext)
    K,F = applyDisplacement( problem, K, F, uspline_bext )
    d = numpy.linalg.solve( K, F )
    d = assembleSolution( d, problem, uspline_bext )
    return d
#=============================================================================================================================================
def assembleSolution( coeff, problem, uspline_bext ):
    disp_node_id = BEXT.getNodeIdNearPoint( uspline_bext, problem[ "displacement" ][ "position" ] )
    coeff = numpy.insert( coeff, disp_node_id, problem[ "displacement" ][ "value" ], axis = 0 )
    return coeff
#=============================================================================================================================================
def applyDisplacement( problem, M, F, uspline_bext ):
    disp_node_id = BEXT.getNodeIdNearPoint( uspline_bext, problem[ "displacement" ][ "position" ] )
    F -= M[:,disp_node_id] * problem[ "displacement" ][ "value" ]
    M = numpy.delete( numpy.delete( M, disp_node_id, axis = 0 ), disp_node_id, axis = 1 )
    F = numpy.delete( F, disp_node_id, axis = 0 )
    return M, F
#=============================================================================================================================================
def applyTraction( problem, force_vector, uspline_bext ):
    elem_id = BEXT.getElementIdContainingPoint( uspline_bext, problem[ "traction" ][ "position" ] )
    elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
    elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
    C = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
    for i in range( 0, elem_degree + 1 ):
        I = elem_nodes[i] 
        Ni = lambda x: Basis.evalSplineBasis1D(x,C,i,elem_domain)
        force_vector[I] += Ni( problem[ "traction" ][ "position" ] ) * problem[ "traction" ][ "value" ]
    return force_vector
#=============================================================================================================================================
def evaluateConstitutiveModel( problem ):
    return problem[ "elastic_modulus" ] * problem[ "area" ]    
#=============================================================================================================================================
def assembleStiffnessMatrix( problem, uspline_bext ):
    basis_deriv = 1
    num_nodes = BEXT.getNumNodes( uspline_bext )
    num_elems = BEXT.getNumElems( uspline_bext )
    M = numpy.zeros( shape = ( num_nodes, num_nodes ) )
    for elem_idx in range( 0, num_elems ):
        elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
        elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
        elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
        elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
        elem_jacobian = ( elem_domain[1] - elem_domain[0] ) / ( 1 - 0 )
        elem_extraction_operator = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
        num_qp = int( numpy.ceil( ( 2*( elem_degree - basis_deriv ) + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1):
            I = elem_nodes[i]
            Ni = lambda x: Basis.evalSplineBasisDeriv1D( elem_extraction_operator, i, basis_deriv, elem_domain, COB.affineMapping(x,[-1,1],elem_domain) )
            for j in range( 0, elem_degree + 1 ):
                J = elem_nodes[j]
                Nj = lambda x: Basis.evalSplineBasisDeriv1D( elem_extraction_operator, j, basis_deriv, elem_domain, COB.affineMapping(x,[-1,1],elem_domain) )
                integrand = lambda x: Ni( x ) * problem[ "elastic_modulus" ] * problem[ "area" ] * Nj( x ) 
                M[I,J] += Quadrature.evaluateGaussLegendreQuadrature(integrand, num_qp, elem_domain) 
    return M
#=============================================================================================================================================
def assembleForceVector( problem, uspline_bext ):
    num_nodes = BEXT.getNumNodes( uspline_bext )
    num_elems = BEXT.getNumElems( uspline_bext )
    force_vector = numpy.zeros( num_nodes )
    for elem_idx in range( 0, num_elems ):
        elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
        elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
        elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
        elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
        C = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
        num_qp = int( numpy.ceil( ( 2*elem_degree + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1 ):
            I = elem_nodes[i] 
            Ni = lambda x: Basis.evalSplineBasis1D(x,C,i,[-1,1])
            integrand = lambda x: Ni( x ) * problem[ "body_force" ]
            force_vector[I] += Quadrature.evaluateGaussLegendreQuadrature(integrand, num_qp, elem_domain) 
    force_vector = applyTraction( problem, force_vector, uspline_bext )
    return force_vector
#=============================================================================================================================================
## UTILITY CODE
def evaluateSolutionAt( x, coeff, uspline_bext ):
    elem_id = BEXT.getElementIdContainingPoint( uspline_bext, x )
    elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
    elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
    elem_extraction_operator = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
    sol = 0.0
    for n in range( 0, len( elem_nodes ) ):
        curr_node = elem_nodes[n] 
        sol += coeff[curr_node] * Basis.evalSplineBasis1D(variate = x , C = elem_extraction_operator , basis_idx = n , domain = elem_domain)
    return sol
#=============================================================================================================================================
def computeElementFitError( problem, coeff, uspline_bext, elem_id ):
    domain = BEXT.getDomain( uspline_bext )
    elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
    num_qp = int( numpy.ceil( ( 2*(elem_degree - 1) + 1 ) / 2.0 ) + 1 )
    abs_err_fun = lambda x : abs( evaluateExactSolutionAt( problem, COB.affineMapping(x,elem_domain,[-1,1]) ) - evaluateSolutionAt( COB.affineMapping(x,elem_domain,[-1,1]), coeff, uspline_bext ) )
    abs_error = Quadrature.evaluateGaussLegendreQuadrature(abs_err_fun, num_qp, elem_domain)    
    return abs_error
#=============================================================================================================================================
def computeFitError( problem, coeff, uspline_bext ):
    num_elems = BEXT.getNumElems( uspline_bext )
    abs_error = 0.0
    for elem_idx in range( 0, num_elems ):
        elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
        abs_error += computeElementFitError( problem, coeff, uspline_bext, elem_id )
    domain = BEXT.getDomain( uspline_bext )
    target_fun_norm, _ = scipy.integrate.quad( lambda x: abs( evaluateExactSolutionAt( problem, x ) ), domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    rel_error = abs_error / target_fun_norm
    return abs_error, rel_error
#=============================================================================================================================================
def plotCompareGoldTestSolution( gold_coeff, test_coeff, uspline_bext ):
    domain = BEXT.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    yg = numpy.zeros( 1000 )
    yt = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        yg[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        yt[i] = evaluateSolutionAt( x[i], gold_coeff, uspline_bext )
    plt.plot( x, yg )
    plt.plot( x, yt )
    plt.show()
#=============================================================================================================================================
def plotCompareFunToExactSolution( problem, test_coeff, uspline_bext ):
    domain = BEXT.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    ya = numpy.zeros( 1000 )
    ye = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        ya[i] = evaluateSolutionAt( x[i], test_coeff, uspline_bext )
        ye[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, ya )
    plt.plot( x, ye )
    plt.show()
#=============================================================================================================================================
def computeConvergenceRate( num_entities, qoi ):
    def func( x, a, b, c ):
        return a * numpy.power( x, b ) + c
    fit = scipy.optimize.curve_fit(func, num_entities, qoi, method='trf', bounds = ([-numpy.inf, -numpy.inf, -numpy.inf ], [numpy.inf, 0.0, numpy.inf]) )
    a,b,c = fit[0]
    return b
#=============================================================================================================================================
def plotSolution( sol_coeff, uspline_bext ):
    domain = BEXT.getDomain( uspline_bext )
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateSolutionAt( x[i], sol_coeff, uspline_bext )
    plt.plot( x, y )
    plt.plot( BEXT.getSplineNodes( uspline_bext )[:,0], sol_coeff, color = "k", marker = "o", markerfacecolor = "k" )
    plt.show()
#=============================================================================================================================================
def evaluateExactSolutionAt( problem, x ):
    term_1 = problem[ "traction" ][ "value" ] / evaluateConstitutiveModel( problem ) * x
    term_2 = problem[ "displacement" ][ "value" ]
    term_3 =  ( ( problem[ "length" ]**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) ) - ( ( ( problem[ "length" ] - x )**2.0 * problem[ "body_force" ] / 2 ) / evaluateConstitutiveModel( problem ) )
    sol = term_1 + term_2 + term_3
    return sol
#=============================================================================================================================================
def plotExactSolution( problem ):
    domain = [0, problem[ "length" ] ]
    x = numpy.linspace( domain[0], domain[1], 1000 )
    y = numpy.zeros( 1000 )
    for i in range(0, len(x) ):
        y[i] = evaluateExactSolutionAt( problem, x[i] )
    plt.plot( x, y )
    plt.show()
#=============================================================================================================================================
class test_ComputeSolution( unittest.TestCase ):
    def test_simple( self ):
           problem = { "elastic_modulus": 100,
                       "area": 0.01,
                       "length": 1.0,
                       "traction": { "value": 1e-3, "position": 1.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 1e-3 }
           spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1, 1 ], "continuity": [ -1, 0, 0, -1 ] }
           Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
           uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
           test_sol_coeff = computeSolution( problem = problem, uspline_bext = uspline_bext )
           gold_sol_coeff = numpy.array( [ 0.0, 11.0 / 18000.0, 1.0 / 900.0, 3.0 / 2000.0 ] )
           self.assertTrue( numpy.allclose( test_sol_coeff, gold_sol_coeff ) )
           # splineBarGalerkin.plotSolution( test_sol_coeff, uspline_bext )
           # splineBarGalerkin.plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )

    def test_textbook_problem( self ):
           problem = { "elastic_modulus": 200e9,
                       "area": 1.0,
                       "length": 5.0,
                       "traction": { "value": 9810.0, "position": 5.0 },
                       "displacement": { "value": 0.0, "position": 0.0 },
                       "body_force": 784800.0 }
           spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
           Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
           uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
           test_sol_coeff = computeSolution( problem = problem, uspline_bext = uspline_bext )
           gold_sol_coeff = numpy.array( [0.0, 2.45863125e-05, 4.92339375e-05, 4.92952500e-05] )
           self.assertTrue( numpy.allclose( test_sol_coeff, gold_sol_coeff ) )
           # splineBarGalerkin.plotSolution( test_sol_coeff, uspline_bext )
           # splineBarGalerkin.plotCompareFunToExactSolution( problem, test_sol_coeff, uspline_bext )