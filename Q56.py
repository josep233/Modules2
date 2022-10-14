import scipy
import Solution
import Mesh
import Basis
import numpy
import matplotlib.pyplot as plt
import math
import sympy

#setup
z = sympy.Symbol('z')
xmin1 = -1
xmax1 = 1
num_elems = 3
# target_fun = lambda x: numpy.sin(numpy.pi * x)
target_fun = lambda x: numpy.e**x
# target_fun = sympy.lambdify(z,sympy.erfc(z))
elem_nodes = [-1,-.33,.33,1]
degree = 2
basis_function = numpy.empty(3)
num = 100
xval = numpy.linspace(xmin1,xmax1,num)
node_coords, ien_array = Mesh.generateMesh1D(xmin1,xmax1,num_elems,degree)
test_solution, node_coords, ien_array = Solution.computeSolution(target_fun,[xmin1,xmax1],num_elems,degree)

#plot boundaries
element_boundaries = numpy.zeros((3,2))
for i in range(0,3):
    element_boundaries1 = Mesh.getElementDomain(i,ien_array,node_coords)
    element_boundaries[i,:] = element_boundaries1
element_boundaries = numpy.ndarray.flatten(element_boundaries)
x = numpy.linspace(-1,1,100)
plt.axhline(0, color='b', xmin=xmin1, xmax=xmax1)
plt.axvline(x=element_boundaries[0], color='b', ymin=xmin1, ymax=xmax1, linestyle=':')
plt.axvline(x=element_boundaries[1], color='b', ymin=xmin1, ymax=xmax1, linestyle=':')
plt.axvline(x=element_boundaries[2], color='b', ymin=xmin1, ymax=xmax1, linestyle=':')
plt.axvline(x=element_boundaries[3], color='b', ymin=xmin1, ymax=xmax1, linestyle=':')
plt.axvline(x=element_boundaries[4], color='b', ymin=xmin1, ymax=xmax1, linestyle=':')
plt.axvline(x=element_boundaries[5], color='b', ymin=xmin1, ymax=xmax1, linestyle=':')

#plot approximation
sol = numpy.empty(num)
for k in range(0,num):
    sol[k] = Solution.evaluateSolutionAt(xval[k], test_solution, node_coords, ien_array, Basis.evaluateLagrangeBasis1D)
plt.plot(xval,sol)

#plot nodal coordinates
plt.plot(node_coords,test_solution,'.')

#plot basis vectors
# x1 = numpy.linspace(.33,1,num)
# vault1 = numpy.zeros(num)
# vaultchanged1 = numpy.zeros(num)
# vaultnorm1 = numpy.zeros(num)
# for i in range(0,num):
#     vault1[i] = Basis.evaluateLagrangeBasis1D(x1[i],degree,2)
#     vaultchanged1[i] = Mesh.spatialToParamCoords(vault1[i],[-1,1])
# plt.plot(x1,vaultchanged1)

#plot properties
plt.xlim([-1,1])
plt.show()
