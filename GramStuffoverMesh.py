import numpy
import Quadrature
import unittest
import Basis
import COB
import Mesh
import scipy
import matplotlib.pyplot as plt
def assembleGramMatrix(node_coords,ien_array,solution_basis):
    num_elems = len(ien_array)
    M = numpy.zeros(shape = (num_elems+1, num_elems+1))
    for elem_idx in range(0,num_elems):
        p = len(ien_array[elem_idx])-1
        omegac = Mesh.getElementDomain(elem_idx,ien_array,node_coords)
        for a in range(0,p+1):
            A = a
            NA = lambda x: solution_basis(x,p,A,[-1,1])
            for b in range(0,p+1):
                B = b
                NB = lambda x: solution_basis(x,p,B,[-1,1])
                integrand = lambda x: NA(x) * NB(x)
                M[A,B] = M[A,B] + Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
    print(M)
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
class Test_assembleGramMatrix( unittest.TestCase ):
    def test_linear_lagrange( self ):
        domain = [ 0, 1 ]
        degree = [ 1, 1 ]
        node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
        test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = Basis.evalLagrangeBasis1D )
        gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
        self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_quadratic_lagrange( self ):
    #     domain = [ 0, 1 ]
    #     degree = [ 2, 2 ]
    #     node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
    #     test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = Basis.evalLagrangeBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1/15, 1/30, -1/60, 0, 0 ], [1/30, 4/15, 1/30, 0, 0], [-1/60, 1/30, 2/15, 1/30, -1/60], [ 0, 0, 1/30, 4/15, 1/30], [0, 0, -1/60, 1/30, 1/15] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    # def test_cubic_lagrange( self ):
    #     domain = [ 0, 1 ]
    #     degree = [ 3, 3 ]
    #     node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
    #     test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = Basis.evalLagrangeBasis1D )
    #     gold_gram_matrix = numpy.array( [ [ 0.03809524,  0.02946429, -0.01071429,  0.00565476,  0.00000000,  0.00000000,  0.00000000 ], 
    #                                       [ 0.02946429,  0.19285714, -0.02410714, -0.01071429,  0.00000000,  0.00000000,  0.00000000 ], 
    #                                       [-0.01071429, -0.02410714,  0.19285714,  0.02946429,  0.00000000,  0.00000000,  0.00000000 ], 
    #                                       [ 0.00565476, -0.01071429,  0.02946429,  0.07619048,  0.02946429, -0.01071429,  0.00565476 ], 
    #                                       [ 0.00000000,  0.00000000,  0.00000000,  0.02946429,  0.19285714, -0.02410714, -0.01071429 ], 
    #                                       [ 0.00000000,  0.00000000,  0.00000000, -0.01071429, -0.02410714,  0.19285714,  0.02946429 ], 
    #                                       [ 0.00000000,  0.00000000,  0.00000000,  0.00565476, -0.01071429,  0.02946429,  0.03809524 ] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    # def test_linear_bernstein( self ):
    #     domain = [ 0, 1 ]
    #     degree = [ 1, 1 ]
    #     node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
    #     test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = Basis.evalBernsteinBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1/6, 1/12, 0], [1/12, 1/3, 1/12], [0, 1/12, 1/6] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )

    # def test_quadratic_bernstein( self ):
    #     domain = [ 0, 1 ]
    #     degree = [ 2, 2 ]
    #     node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
    #     test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = Basis.evalBernsteinBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1/10, 1/20, 1/60, 0, 0 ], [1/20, 1/15, 1/20, 0, 0 ], [1/60, 1/20, 1/5, 1/20, 1/60], [0, 0, 1/20, 1/15, 1/20], [0, 0, 1/60, 1/20, 1/10] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )
    
    # def test_cubic_bernstein( self ):
    #     domain = [ 0, 1 ]
    #     degree = [ 3, 3 ]
    #     node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
    #     test_gram_matrix = assembleGramMatrix( node_coords = node_coords, ien_array = ien_array, solution_basis = Basis.evalBernsteinBasis1D )
    #     gold_gram_matrix = numpy.array( [ [1/14, 1/28, 1/70, 1/280, 0, 0, 0 ], [1/28, 3/70, 9/280, 1/70, 0, 0, 0 ], [1/70, 9/280, 3/70, 1/28, 0, 0, 0 ], [1/280, 1/70, 1/28, 1/7, 1/28, 1/70, 1/280], [0, 0, 0, 1/28, 3/70, 9/280, 1/70], [0, 0, 0, 1/70, 9/280, 3/70, 1/28], [0, 0, 0, 1/280, 1/70, 1/28, 1/14 ] ] )
    #     self.assertTrue( numpy.allclose( test_gram_matrix, gold_gram_matrix ) )