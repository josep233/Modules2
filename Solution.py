import sympy
import unittest
import math
import numpy
import Mesh
import Basis
import COB
import BEXT

def computeSolution(target_fun,domain,num_elems,degree):
    node_coords,ien_array = Mesh.generateMesh1D(domain[0],domain[-1],num_elems,degree)
    test_solution = target_fun(node_coords)
    return test_solution, node_coords, ien_array
##==============================================================================================================
def evaluateSolutionAt(x, coeff, node_coords, ien_array, eval_basis):
    elem_idx = Mesh.getElementIdxContainingPoint(x,node_coords, ien_array)
    elem_nodes = Mesh.getElementNodes(elem_idx,ien_array)
    elem_domain = Mesh.getElementDomain(elem_idx,ien_array,node_coords)
    param_coord = Mesh.spatialToParamCoords(x,elem_domain)

    sol_at_point = 0
    for i in range(0,len(elem_nodes)):
        curr_node = elem_nodes[i]
        degree = len(ien_array[0])-1

        sol_at_point += coeff[curr_node] * eval_basis(param_coord,degree,i)
    return sol_at_point
##==============================================================================================================
# class Test_computeSolution( unittest.TestCase ):
#     def test_single_linear_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x, domain = [-1.0, 1.0 ], num_elems = 1, degree = 1 )
#         gold_solution = numpy.array( [ -1.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
#     def test_single_quad_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 1, degree = 2 )
#         gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
#     def test_two_linear_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 2, degree = 1 )
#         gold_solution = numpy.array( [ 1.0, 0.0, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
#     def test_four_quad_element_poly( self ):
#         test_solution, node_coords, ien_array = computeSolution( target_fun = lambda x : x**2, domain = [-1.0, 1.0 ], num_elems = 4, degree = 1 )
#         gold_solution = numpy.array( [ 1.0, 0.25, 0.0, 0.25, 1.0 ] )
#         self.assertTrue( numpy.allclose( test_solution, gold_solution ) )
##==============================================================================================================
class Test_evaluateSolutionAt( unittest.TestCase ):
    def test_single_linear_element( self ):
        node_coords, ien_array = Mesh.generateMesh1D( -1, 1, 1, 1 )
        coeff = numpy.array( [-1.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = -1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.0 )
    def test_two_linear_elements( self ):
        node_coords, ien_array = Mesh.generateMesh1D( -1, 1, 2, 1 )
        coeff = numpy.array( [ 1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.0 )
    def test_single_quadratic_element( self ):
        node_coords, ien_array = Mesh.generateMesh1D( -1, 1, 1, 2 )
        coeff = numpy.array( [+1.0, 0.0, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second =  0.0 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.0 )
    def test_two_quadratic_elements( self ):
        node_coords, ien_array = Mesh.generateMesh1D( -2, 2, 2, 2 )
        coeff = numpy.array( [ 1.0, 0.25, 0.5, 0.25, 1.0 ] )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.00 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = -1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x =  0.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +0.50 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +1.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +0.25 )
        self.assertAlmostEqual( first = evaluateSolutionAt( x = +2.0, coeff = coeff, node_coords = node_coords, ien_array = ien_array, eval_basis = Basis.evaluateLagrangeBasis1D ), second = +1.00 )