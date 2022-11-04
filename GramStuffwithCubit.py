import numpy
import Quadrature
import unittest
import Basis
import COB
import Mesh
import scipy
import matplotlib.pyplot as plt
import BEXT

def assembleGramMatrix(uspline):
    num_elems = BEXT.getNumElems( uspline )
    num_nodes = BEXT.getNumNodes( uspline )
    M = numpy.zeros(shape = (num_nodes, num_nodes))
    for elem_id in range(0,num_elems):
        p = BEXT.getElementDegree( uspline, elem_id )
        omegac = BEXT.getElementDomain( uspline, elem_id )
        C = BEXT.getElementExtractionOperator( uspline, elem_id )
        for a in range(0,p+1):
            A = BEXT.getElementIdContainingPoint( uspline, a )
            NA = lambda x: Basis.evalBernsteinBasis1D(x,p,a,[-1,1])
            for b in range(0,p+1):
                B = BEXT.getElementIdContainingPoint( uspline, a )
                NB = lambda x: Basis.evalBernsteinBasis1D(x,p,b,[-1,1])
                integrand = lambda x: NA(x) * NB(x)
                M[A,B] = M[A,B] + Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
    return M
#=============================================================================================================================================
def assembleForceVector(target_fun,node_coords,ien_array,solution_basis):
  num_elems = len(ien_array)
  num_nodes = len(node_coords)
  F = numpy.zeros(num_nodes)
  for elem_idx in range(0,num_elems):
    p = len(ien_array[elem_idx])-1
    omegac = Mesh.getElementDomain(elem_idx,ien_array,node_coords)
    for a in range(0,p+1):
      A = ien_array[elem_idx][a]
      NA = lambda x: solution_basis(x,p,a,[-1,1])
      integrand = lambda x: NA(x) * target_fun(COB.affineMapping(x,[-1,1],omegac))
      F[A] += Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
  return F
#=============================================================================================================================================
def computeGalerkinApproximation(target_fun,domain,degree,solution_basis):
  node_coords, ien_array = Mesh.generateMesh1D( domain[0], domain[1], degree )
  M = assembleGramMatrix(node_coords,ien_array,solution_basis)
  F = assembleForceVector(target_fun,node_coords,ien_array,solution_basis)
  d = numpy.linalg.inv(M) @ F
  return d, node_coords,ien_array
