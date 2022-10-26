# first line: 30
@joblib.Memory("cachedir").cache()
def evalLagrangeBasis1D(variate,degree,basis_idx,domain_in):
    domain_out = [-1,1]
    variate = COB.affineMapping(variate,domain_in,domain_out)
    z = sympy.Symbol('z')
    P = 1
    xj = numpy.linspace(-1,1,degree+1)
    for i in range(0,degree+1):
        if basis_idx != i:
            P *= (z - xj[i]) / (xj[basis_idx] - xj[i])
    ans = P.subs(z,variate)
    return ans
