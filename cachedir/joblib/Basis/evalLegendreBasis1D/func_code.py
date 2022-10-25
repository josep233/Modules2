# first line: 13
@joblib.Memory("cachedir").cache()
def evalLegendreBasis1D(variate,degree,basis_idx,domain_in):
    domain_out = [-1,1]
    variate = COB.affineMapping(variate,domain_in,domain_out)
    z = sympy.Symbol('z')
    legendrefunc = 1/((2**basis_idx)*math.factorial(basis_idx)) * sympy.diff((z**2 - 1)**basis_idx,z,basis_idx)
    return legendrefunc.subs(z,variate)
