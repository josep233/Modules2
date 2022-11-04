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
    for elem_idx in range(0,num_elems):
        elem_id = BEXT.elemIdFromElemIdx( uspline, elem_idx )
        p = BEXT.getElementDegree( uspline, elem_id )
        omegac = BEXT.getElementDomain( uspline, elem_id )
        elem_nodes = BEXT.getElementNodeIds( uspline, elem_id )
        C = BEXT.getElementExtractionOperator( uspline, elem_id )
        for a in range(0,p+1):
            A = elem_nodes[a]
            NA = lambda x: Basis.evalSplineBasis1D(x,C,a,[-1,1])
            for b in range(0,p+1):
                B = elem_nodes[b]
                NB = lambda x: Basis.evalSplineBasis1D(x,C,a,[-1,1])
                integrand = lambda x: NA(x) * NB(x)
                M[A,B] += Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
    return M
#=============================================================================================================================================
def assembleForceVector(target_fun,uspline):
  num_elems = BEXT.getNumElems( uspline )
  num_nodes = BEXT.getNumNodes( uspline )
  F = numpy.zeros(num_nodes)
  for elem_idx in range(0,num_elems):
    elem_id = BEXT.elemIdFromElemIdx( uspline, elem_idx )
    p = BEXT.getElementDegree( uspline, elem_id )
    omegac = BEXT.getElementDomain( uspline, elem_id )
    elem_nodes = BEXT.getElementNodeIds( uspline, elem_id )
    C = BEXT.getElementExtractionOperator( uspline, elem_id )
    for a in range(0,p+1):
      A = elem_nodes[a]
      A = BEXT.getElementIdContainingPoint( uspline, a )
      NA = lambda x: Basis.evalSplineBasis1D(x,C,a,[-1,1])
      integrand = lambda x: NA(x) * target_fun(COB.affineMapping(x,[-1,1],omegac))
      F[A] += Quadrature.evaluateGaussLegendreQuadrature(integrand, int(numpy.ceil((2 * p + 1) / 2)), omegac)
  return F
#=============================================================================================================================================
def computeSolution( target_fun, uspline ):
    M = assembleGramMatrix( uspline )
    F = assembleForceVector( target_fun, uspline )
    d = numpy.linalg.solve( M, F )
    return d