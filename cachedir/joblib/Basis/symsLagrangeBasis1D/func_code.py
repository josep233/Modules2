# first line: 48
@joblib.Memory("cachedir").cache()
def symsLagrangeBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    P = 1
    xj = numpy.linspace(-1,1,degree+1)
    for i in range(0,degree+1):
        if basis_idx != i:
            P *= (z - xj[i]) / (xj[basis_idx] - xj[i])
    return P
