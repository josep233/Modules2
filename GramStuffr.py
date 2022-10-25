import numpy
import Quadrature

def assembleGramMatrix(degree,eval_basis,domain):
  M = numpy.zeros(shape = (degree + 1, degree + 1))
  for A in range(0,degree + 1):
    NA = lambda x: eval_basis( degree, A, x)
    for B in range(0, degree + 1):
      NB = lambda x: eval_basis( degree, B, x)
      integrand = lambda x: NA(x) * NB(x)
      M[A,B] = Quadrature.quad( integrand, domain)
  return M

#issues with this code: eval_basis and Quadrature functions operate over a certain domain. COB is needed to correctly integrate or approximate.

def assembleForceVector(force_function,degree,eval_basis,domain):
  F = numpy.zeros(degree + 1)
  for A in range(0,degree + 1):
    NA = lambda x: eval_basis( degree, A, x)
    integrand = lambda x: NA(x) * force_function(basis.affine_mapping(domain,[0,1],x))
    F[A] = Quadrature.quad( integrand, domain)
  return F

#same problems as above

#for all of the above functions, when using quadrature, you are integrating two functions with power p. Thus, the degree of quadrature must be p + p. 