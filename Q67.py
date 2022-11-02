import numpy
import BEXT
import Basis
import matplotlib
import Solution

def evaluateElementBernsteinBasisAtParamCoord( uspline, elem_id, param_coord ):
    elem_degree = BEXT.getElementDegree( uspline, elem_id )
    elem_bernstein_basis = numpy.zeros( elem_degree + 1 )
    for n in range( 0, elem_degree + 1 ):
        elem_bernstein_basis[n] = Basis.evalBernsteinBasis1D(variate,elem_degree,basis_idx,param_coord)
    return elem_bernstein_basis

def evaluateElementSplineBasisAtParamCoord( uspline, elem_id, param_coord ):
    elem_ext_operator = BEXT.getElementExtractionOperator( uspline, elem_id )
    elem_bernstein_basis = evaluateElementBernsteinBasisAtParamCoord( uspline, elem_id, param_coord )
    elem_spline_basis = elem_ext_operator * elem_bernstein_basis
    return elem_spline_basis 

def plotUsplineBasis( uspline, color_by ):
    elem_degree = BEXT.getElementDegree( uspline, elem_id )
    num_pts = 100
    xi = numpy.linspace( 0, 1, num_pts )
    num_elems = BEXT.getNumElems( uspline )
    for elem_idx in range( 0, num_elems ):
        elem_id = BEXT.elemIdFromElemIdx( uspline, elem_idx )
        elem_domain = BEXT.getElementDomain( uspline, elem_id )
        elem_node_ids = BEXT.getElementNodeIds( uspline, elem_id )
        x = numpy.linspace( elem_domain[0], elem_domain[1], num_pts )
        y = numpy.zeros( shape = ( elem_degree + 1, num_pts ) )
        for i in range( 0, num_pts ):
            y[:,i] = Solution.evaluateSplineBasisAtPoint(uspline, node_coords, ien_array, x[i])
        # Do plotting for the current element
        for n in range( 0, elem_degree + 1 ):
            if color_by == "element":
                color = getLineColor( elem_idx )
            elif color_by == "node":
                color = getLineColor( elem_node_ids[n] )
            matplotlib.pyplot.plot( x, y[n,:], color = getLineColor( color ) )
    matplotlib.pyplot.show()

def getLineColor( idx ):
    colors = list( matplotlib.colors.TABLEAU_COLORS.keys() )
    num_colors = len( colors )
    mod_idx = idx % num_colors
    return matplotlib.colors.TABLEAU_COLORS[ colors[ mod_idx ] ]

uspline = BEXT.readBEXT( "your_uspline_bext.json" )
plotUsplineBasis( uspline, "element" )
plotUsplineBasis( uspline, "node" )