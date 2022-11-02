import sympy
import unittest
import math
import numpy
import COB
import Basis

def generateMesh1D(xmin,xmax,degree):
    ien_array = {}
    num_elems =  len(degree)
    elem_boundaries = numpy.linspace(xmin,xmax,num_elems+1)
    init = 0
    node_coords = []
    for elem_id in range(0,num_elems):
        elem_node_coords = numpy.linspace(elem_boundaries[elem_id],elem_boundaries[elem_id+1],degree[elem_id]+1)
        if elem_id == 0:
            node_coords.append(elem_node_coords)
        else:
            node_coords.append(elem_node_coords[1:])
        ien_array[elem_id] = numpy.arange(init,init+degree[elem_id]+1).tolist()
        init = ien_array[elem_id][-1]
    node_coords = numpy.concatenate(node_coords)
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
def spatialToParamCoords(x,old_domain,new_domain):
    b1 = numpy.array(new_domain)
    b2 = numpy.array(old_domain)
    constant = (b1[-1] - b1[0]) / (b2[-1] - b2[0])
    b2tild = b2 * constant
    shift = b2tild[0] - b1[0]
    param_coord = x*constant - shift
    return param_coord
# #=====================================================================================================
def getGlobalNodeID(ien_array,node_coords,elem):
    
    return
# #=====================================================================================================
class Test_getGlobalNodeID( unittest.TestCase ):
    def test_globalID1( self ):
        degree = [1]
        elem = 0
        gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = degree )
        testglobalID = getGlobalNodeID(node_coords, ien_array, elem)
        goldglobalID = 0
        self.assertAlmostEqual(first=testglobalID,second=goldglobalID)

# #=====================================================================================================
class Test_generateMesh1D( unittest.TestCase ):
    def test_make_1_linear_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = [ 1 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    def test_make_1_quadratic_elem( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1, 2 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = [ 2 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    def test_make_2_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.5, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ], 1: [ 1, 2 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = [ 1, 1 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    def test_make_2_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1, 2 ], 1: [ 2, 3, 4 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = [ 2, 2 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    def test_make_4_linear_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.25, 0.5, 0.75, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1 ], 1: [ 1, 2 ], 2: [ 2, 3 ], 3: [ 3, 4 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = [ 1, 1, 1, 1 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    def test_make_4_quadratic_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 ] )
        gold_ien_array = { 0: [ 0, 1, 2 ], 1: [ 2, 3, 4 ], 2: [ 4, 5, 6 ], 3: [ 6, 7, 8 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 1.0, degree = [ 2, 2, 2, 2 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
    def test_make_4_p_refine_elems( self ):
        gold_node_coords = numpy.array( [ 0.0, 1.0, 1.5, 2.0, (2.0 + 1.0/3.0), (2.0 + 2.0/3.0), 3.0, 3.25, 3.5, 3.75, 4.0 ] )
        gold_ien_array = { 0: [ 0, 1 ], 1: [ 1, 2, 3 ], 2: [ 3, 4, 5, 6 ], 3: [ 6, 7, 8, 9, 10 ] }
        node_coords, ien_array = generateMesh1D( xmin = 0.0, xmax = 4.0, degree = [ 1, 2, 3, 4 ] )
        self.assertIsInstance( obj = node_coords, cls = numpy.ndarray )
        self.assertIsInstance( obj = ien_array, cls = dict )
        self.assertTrue( numpy.allclose( node_coords, gold_node_coords ) )
        self.assertDictEqual( d1 = gold_ien_array, d2 = ien_array )
#=====================================================================================================
class Test_getElementDomain( unittest.TestCase ):
    def test_linear_1_elem_domain( self ):
        v1=getElementDomain(0,[[0,1]],numpy.array([-1,1]))
        v2=numpy.array([-1,1])
        self.assertTrue( numpy.allclose( v1, v2 ) )
    def test_linear_2_elem_domain( self ):
        v1=getElementDomain(0,[[0,1],[1,2]],numpy.array([-1,0,1]))
        v2=numpy.array([-1,0])
        self.assertTrue( numpy.allclose( v1, v2 ) )
    def test_linear_2_elem_domain2( self ):
        v1=getElementDomain(1,[[0,1],[1,2]],numpy.array([-1,0,1]))
        v2=numpy.array([0,1])
        self.assertTrue( numpy.allclose( v1, v2 ) )
    def test_quadratic_1_elem_domain( self ):
        v1=getElementDomain(0,[[0,1,2]],numpy.array([-1,0,1]))
        v2=numpy.array([-1,1])
        self.assertTrue( numpy.allclose( v1, v2 ) )
    def test_quadratic_1_elem_domain( self ):
        v1=getElementDomain(0,[[0,1,2],[2,3,4]],numpy.array([-1,-.5,0,.5,1]))
        v2=numpy.array([-1,0])
        self.assertTrue( numpy.allclose( v1, v2 ) )
    def test_quadratic_1_elem_domain( self ):
        v1=getElementDomain(1,[[0,1,2],[2,3,4]],numpy.array([-1,-.5,0,.5,1]))
        v2=numpy.array([0,1])
        self.assertTrue( numpy.allclose( v1, v2 ) )
#=====================================================================================================
class Test_getElementNodes( unittest.TestCase ):
    def test_linear_1_elem_nodes( self ):
        v1=getElementNodes(0,[[0,1]])
        v2=numpy.array([0,1])
        self.assertTrue( numpy.allclose( v1, v2 ) )
    def test_linear_2_elem_nodes( self ):
        v1=getElementNodes(0,[[0,1],[1,2]])
        v2=numpy.array([0,1])
        self.assertTrue( numpy.allclose( v1, v2 ) )
        v1=getElementNodes(1,[[0,1],[1,2]])
        v2=numpy.array([1,2])
        self.assertTrue( numpy.allclose( v1, v2 ) ) 
    def test_quadratic_1_elem_nodes( self ):
        v1=getElementNodes(0,[[0,1,2]])
        v2=numpy.array([0,1,2]) 
    def test_quadratic_2_elem_nodes( self ):
        v1=getElementNodes(0,[[0,1,2],[2,3,4]])
        v2=numpy.array([0,1,2])
        self.assertTrue( numpy.allclose( v1, v2 ) )
        v1=getElementNodes(1,[[0,1,2],[2,3,4]])
        v2=numpy.array([2,3,4])
        self.assertTrue( numpy.allclose( v1, v2 ) )
#=====================================================================================================
class Test_spatialToParamCoords( unittest.TestCase ):
    def test_unit_to_param_domain( self ):
        self.assertAlmostEqual(first=spatialToParamCoords(0,[0,1],[-1,1]),second=-1)
        self.assertAlmostEqual(first=spatialToParamCoords(.5,[0,1],[-1,1]),second=0)
        self.assertAlmostEqual(first=spatialToParamCoords(1,[0,1],[-1,1]),second=1)
#=====================================================================================================
class Test_getElementIdxContainingPoint( unittest.TestCase ):
    def test_linear_1_elem_idx( self ):
        self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,1], [[0,1]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(0,[-1,1], [[0,1]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,1], [[0,1]]),second=0)
    def test_linear_2_elem_idx( self ):
        self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,0,1], [[0,1],[1,2]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(-.5,[-1,0,1], [[0,1],[1,2]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(.5,[-1,0,1], [[0,1],[1,2]]),second=1)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,0,1], [[0,1],[1,2]]),second=1)
    def test_quadratic_1_elem_idx( self ):
        self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,0,1], [[0,1,2]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(0,[-1,0,1], [[0,1,2]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,0,1], [[0,1,2]]),second=0)
    def test_quadratic_2_elem_idx( self ):
        self.assertAlmostEqual(first=getElementIdxContainingPoint(-1,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(-.5,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=0)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(.5,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=1)
        self.assertAlmostEqual(first=getElementIdxContainingPoint(1,[-1,-.5,0,.5,1], [[0,1,2],[2,3,4]]),second=1)
