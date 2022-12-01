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

#=============================================================================================================================================
## SECONDARY FUNCTIONS
# def computeSolution( target_fun, uspline_bext ):
#     M = assembleGramMatrix( uspline_bext )
#     F = assembleForceVector( target_fun, uspline_bext )
#     d = numpy.linalg.solve( M, F )
#     return d
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
def assembleForceVector(problem,uspline_bext):
    basis_deriv = 1
    num_nodes = BEXT.getNumNodes( uspline_bext )
    num_elems = BEXT.getNumElems( uspline_bext )
    force_vector = numpy.zeros( shape = ( num_nodes, 1 ) )
    for elem_idx in range( 0, num_elems ):
        elem_id = BEXT.elemIdFromElemIdx( uspline_bext, elem_idx )
        elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
        elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
        elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
        elem_jacobian = ( elem_domain[1] - elem_domain[0] ) / ( 1 - 0 )
        elem_extraction_operator = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
        num_qp = int( numpy.ceil( ( 2*( elem_degree - basis_deriv ) + 1 ) / 2.0 ) )
        for i in range( 0, elem_degree + 1 ):
            I = elem_nodes[i]
            Ni = lambda x: Basis.evalSplineBasisDeriv1D( elem_extraction_operator, i, basis_deriv, elem_domain, COB.affineMapping(x,[-1,1],elem_domain) )
            integrand = lambda x: Ni( x ) * problem["body_force"] * Ni( x ) 
            force_vector[I] += Quadrature.evaluateGaussLegendreQuadrature(integrand, num_qp, elem_domain) 
    force_vector = applyTraction( problem, force_vector, uspline_bext )
    return force_vector
#=============================================================================================================================================
def applyTraction( problem, force_vector, uspline_bext ):
    elem_id = BEXT.getNodeIdNearPoint( uspline_bext, problem["traction"]["position"] )
    elem_domain = BEXT.getElementDomain( uspline_bext, elem_id )
    elem_degree = BEXT.getElementDegree( uspline_bext, elem_id )
    elem_nodes = BEXT.getElementNodeIds( uspline_bext, elem_id )
    C = BEXT.getElementExtractionOperator( uspline_bext, elem_id )
    for i in range( 0, elem_degree + 1 ):
        I = elem_nodes[i]
        Ni = lambda x: Basis.evalSplineBasis1D( x, C, i, elem_domain )
        force_vector[I] += Ni( problem[ "traction" ][ "position" ] ) * problem[ "traction" ][ "value" ]
    return force_vector
#=============================================================================================================================================
def applyDisplacement( problem, M, force_vector, uspline_bext ):
    elem_id = BEXT.getNodeIdNearPoint( uspline_bext, problem["displacement"]["position"] )
    force_vector -= M[:,elem_id] * problem[ "displacement" ][ "value" ]
    M = numpy.delete( numpy.delete( M, elem_id, axis = 0 ), elem_id, axis = 1 )
    force_vector = numpy.delete( force_vector, elem_id, axis = 0 )
    return M, force_vector
#=============================================================================================================================================
def computeBarGalerkinApproximation(problem, uspline_bext):
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
class test_assembleStressMatrix( unittest.TestCase ):
       def test_one_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1 ], "continuity": [ -1, -1 ] }
              Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 1.0, -1.0 ], [ -1.0, 1.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_linear_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 1, 1 ], "continuity": [ -1, 0, -1 ] }
              Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [ 2.0, -2.0, 0.0 ], [ -2.0, 4.0, -2.0 ], [ 0.0, -2.0, 2.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_one_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2 ], "continuity": [ -1, -1 ] }
              Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  4.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0,  4.0 / 3.0, -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0, -2.0 / 3.0,  4.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_quadratic_C0_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 0, -1 ] }
              Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                 [ -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0,  0.0,        0.0 ],
                                                 [ -4.0 / 3.0, -4.0 / 3.0, 16.0 / 3.0, -4.0 / 3.0, -4.0 / 3.0 ],
                                                 [  0.0,        0.0,       -4.0 / 3.0,  8.0 / 3.0, -4.0 / 3.0 ],
                                                 [  0.0,        0.0,       -4.0 / 3.0, -4.0 / 3.0,  8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )

       def test_two_quadratic_C1_element( self ):
              problem = { "elastic_modulus": 100,
                     "area": 0.01,
                     "length": 1.0,
                     "traction": { "value": 1e-3, "position": 1.0 },
                     "displacement": { "value": 0.0, "position": 0.0 },
                     "body_force": 0.0 }
              spline_space = { "domain": [0, problem[ "length" ]], "degree": [ 2, 2 ], "continuity": [ -1, 1, -1 ] }
              Uspline.make_uspline_mesh( spline_space, "temp_uspline" )
              uspline_bext = BEXT.readBEXT( "temp_uspline.json" )
              test_stiffness_matrix = assembleStiffnessMatrix( problem = problem, uspline_bext = uspline_bext )
              gold_stiffness_matrix = numpy.array( [ [  8.0 / 3.0, -2.0,       -2.0/ 3.0,   0.0 ],
                                                 [ -2.0,        8.0 / 3.0,  0.0,       -2.0 / 3.0 ],
                                                 [ -2.0 / 3.0,  0.0,        8.0 / 3.0, -2.0 ],
                                                 [  0.0,       -2.0 / 3.0, -2.0,        8.0 / 3.0 ] ] )
              self.assertTrue( numpy.allclose( test_stiffness_matrix, gold_stiffness_matrix ) )