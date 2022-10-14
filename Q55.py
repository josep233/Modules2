import scipy
import Solution
import Mesh
import Basis
import numpy
import matplotlib.pyplot as plt
import math

def computeFitError( target_fun, coeff, node_coords, ien_array, eval_basis ):
    num_elems = ien_array.shape[0]
    domain = [ min( node_coords ), max( node_coords ) ]
    abs_err_fun = lambda x : abs( target_fun( x ) - Solution.evaluateSolutionAt( x, coeff, node_coords, ien_array, eval_basis ) )
    fit_error, residual = scipy.integrate.quad( abs_err_fun, domain[0], domain[1], epsrel = 1e-12, limit = num_elems * 100 )
    return fit_error, residual

k = 5
error = numpy.zeros(k)
nodes = numpy.zeros(k)
elems = numpy.linspace(1,10**3,10)
elems = elems.astype(int)

for i in range(1,k+1):
    xmin = -1
    xmax = 1
    domain = numpy.array([xmin, xmax])
    num_elems = elems[i-1]
    degree = 1

    target_fun = lambda x: numpy.sin(3.141 * x)
    coeff,node_coords,ien_array = Solution.computeSolution(target_fun,domain,num_elems,degree)
    eval_basis = Basis.evaluateLagrangeBasis1D

    fit_error,residual = computeFitError( target_fun, coeff, node_coords, ien_array, eval_basis )
    error[i-1] = fit_error
    nodes[i-1] = len(node_coords)

print(error)
print(nodes)

# y = [3.12500000e-02, 2.63352630e-08, 2.83182204e-09, 8.40326917e-10, 3.17559956e-10, 1.88367666e-10, 8.16511459e-11, 6.33578793e-11, 5.10710862e-11, 3.39643491e-11]
# x = elems
# x = [   3.,  225.,  447.,  669.,  891., 1113., 1335., 1557., 1779., 2001.]

# plt.loglog(x,y,".")
# plt.xlabel('Number of Nodes')
# plt.ylabel('Error')
# plt.show()