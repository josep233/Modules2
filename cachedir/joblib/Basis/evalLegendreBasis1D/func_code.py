# first line: 13
@joblib.Memory("cachedir").cache()
def evalLegendreBasis1D(variate,degree,basis_idx,space_domain):
    param_domain = [-1,1]
    variate = COB.affineMapping(variate,space_domain,param_domain)
    z = sympy.Symbol('z')
    legendrefunc = 1/((2**basis_idx)*math.factorial(basis_idx)) * sympy.diff((z**2 - 1)**basis_idx,z,basis_idx)
    return legendrefunc.subs(z,variate)
