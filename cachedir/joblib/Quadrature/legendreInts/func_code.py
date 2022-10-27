# first line: 55
@joblib.Memory("cachedir").cache()
def legendreInts(n,domain):
    # z = sympy.Symbol('z')
    # M = numpy.zeros(2*n, dtype = "double")
    # for i in range(0,len(M)):
    #     fun = Basis.evalLegendreBasis1D(z,n,i,domain)
    #     M[i] = sympy.integrate(fun,(z,domain[0],domain[-1]))
    M = numpy.zeros(2*n, dtype = "double")
    M[0] = 2.0
    return M
