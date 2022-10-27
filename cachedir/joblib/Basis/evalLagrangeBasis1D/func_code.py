# first line: 39
@joblib.Memory("cachedir").cache()
def evalLagrangeBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [-1,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    z = sympy.Symbol('z')
    P = 1
    xj = numpy.linspace(-1,1,degree+1)
    P = symsLagrangeBasis1D(variate,degree,basis_idx,space_domain)
    ans = P.subs(z,ans)
    return ans
