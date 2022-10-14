from ast import increment_lineno
from re import A
import sympy
import unittest
import math
import numpy
import COB
import Basis

def generateMesh1D(xmin,xmax,num_elems,degree):
    node_coords = numpy.linspace(xmin,xmax,degree+1+(num_elems-1)*degree)
    ien_array = numpy.zeros([num_elems,degree+1],dtype="int")
    for i in range(num_elems):
        if i != 0:
            startindex = ien_array[i-1,-1]
        else:
            startindex = 0
        ien_array[i] = numpy.arange(startindex,startindex+degree+1)        
    return node_coords,ien_array
##=====================================================================================================
def getElementIdxContainingPoint(x,node_coords, ien_array):
        for i in range(0,len(ien_array)):
            if float(x) >= node_coords[ien_array[i][0]] and float(x) <= node_coords[ien_array[i][-1]]:
                elem_idx = i
        return elem_idx
##=====================================================================================================
def getElementNodes(elem_idx,ien_array):
    elem_nodes = ien_array[elem_idx][:]
    return elem_nodes
##=====================================================================================================
def getElementDomain(elem_idx,ien_array,node_coords):
    elem_domain = numpy.array([node_coords[ien_array[elem_idx][0]],node_coords[ien_array[elem_idx][-1]]])
    return elem_domain
##=====================================================================================================
def spatialToParamCoords(x,elem_domain):
    b1 = numpy.array([-1,1])
    b2 = numpy.array(elem_domain)
    constant = (b1[-1] - b1[0]) / (b2[-1] - b2[0])
    b2tild = b2 * constant
    shift = b2tild[0] - b1[0]
    param_coord = x*constant - shift
    return param_coord
# #=====================================================================================================
# class Test_getElementDomain( unittest.TestCase ):
#     def test_linear_1_elem_domain( self ):
#         v1=getElementDomain(0,[[0,1]],numpy.array([-1,1]))
#         v2=numpy.array([-1,1])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#     def test_linear_2_elem_domain( self ):
#         v1=getElementDomain(0,[[0,1],[1,2]],numpy.array([-1,0,1]))
#         v2=numpy.array([-1,0])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#     def test_linear_2_elem_domain2( self ):
#         v1=getElementDomain(1,[[0,1],[1,2]],numpy.array([-1,0,1]))
#         v2=numpy.array([0,1])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#     def test_quadratic_1_elem_domain( self ):
#         v1=getElementDomain(0,[[0,1,2]],numpy.array([-1,0,1]))
#         v2=numpy.array([-1,1])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#     def test_quadratic_1_elem_domain( self ):
#         v1=getElementDomain(0,[[0,1,2],[2,3,4]],numpy.array([-1,-.5,0,.5,1]))
#         v2=numpy.array([-1,0])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#     def test_quadratic_1_elem_domain( self ):
#         v1=getElementDomain(1,[[0,1,2],[2,3,4]],numpy.array([-1,-.5,0,.5,1]))
#         v2=numpy.array([0,1])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
# #=====================================================================================================
# class Test_getElementNodes( unittest.TestCase ):
#     def test_linear_1_elem_nodes( self ):
#         v1=getElementNodes(0,[[0,1]])
#         v2=numpy.array([0,1])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#     def test_linear_2_elem_nodes( self ):
#         v1=getElementNodes(0,[[0,1],[1,2]])
#         v2=numpy.array([0,1])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#         v1=getElementNodes(1,[[0,1],[1,2]])
#         v2=numpy.array([1,2])
#         self.assertTrue( numpy.allclose( v1, v2 ) ) 
#     def test_quadratic_1_elem_nodes( self ):
#         v1=getElementNodes(0,[[0,1,2]])
#         v2=numpy.array([0,1,2]) 
#     def test_quadratic_2_elem_nodes( self ):
#         v1=getElementNodes(0,[[0,1,2],[2,3,4]])
#         v2=numpy.array([0,1,2])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
#         v1=getElementNodes(1,[[0,1,2],[2,3,4]])
#         v2=numpy.array([2,3,4])
#         self.assertTrue( numpy.allclose( v1, v2 ) )
# #=====================================================================================================
# class Test_spatialToParamCoords( unittest.TestCase ):
#     def test_unit_to_param_domain( self ):
#         self.assertAlmostEqual(first=spatialToParamCoords(0,[0,1]),second=-1)
#         self.assertAlmostEqual(first=spatialToParamCoords(.5,[0,1]),second=0)
#         self.assertAlmostEqual(first=spatialToParamCoords(1,[0,1]),second=1)
# #=====================================================================================================
# class Test_getElementIdxContainingPoint( unittest.TestCase ):
#     def test_linear_1_elem_idx( self ):
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,1], [[0,1]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(0,[-1,1], [[0,1]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,1], [[0,1]]),second=0)
#     def test_linear_2_elem_idx( self ):
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,0,1], [[0,1],[1,2]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(-.5,[-1,0,1], [[0,1],[1,2]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(.5,[-1,0,1], [[0,1],[1,2]]),second=1)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,0,1], [[0,1],[1,2]]),second=1)
#     def test_quadratic_1_elem_idx( self ):
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,0,1], [[0,1,2]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(0,[-1,0,1], [[0,1,2]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,0,1], [[0,1,2]]),second=0)
#     def test_quadratic_2_elem_idx( self ):
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(-.5,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=0)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(.5,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=1)
#         self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=1)
# #=====================================================================================================
# class Test_generateMesh1D( unittest.TestCase ):
#     def test_make_1_linear_elem( self ):
#         gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 1 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
#     def test_make_2_linear_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 1 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
#     def test_make_4_linear_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 3, 4 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 1 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
#     def test_make_1_quadratic_elem( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1, 2 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 1, degree = 2 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
#     def test_make_2_quadratic_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 2, degree = 2 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )
#     def test_make_4_quadratic_elems( self ):
#         gold_node_coords = numpy.array( [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ] )
#         gold_ien_array = numpy.array( [ [ 0, 1, 2 ], [ 2, 3, 4 ], [ 4, 5, 6 ], [ 6, 7, 8 ] ], dtype = int )
#         node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, num_elems = 4, degree = 2 )
#         self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
#         self.assertIsInstance( obj = ien_array, cls = numpy.ndarray )
#         self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
#         self.assertTrue( numpy.array_equiv( ien_array, gold_ien_array ) )