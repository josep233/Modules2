# first line: 19
@joblib.Memory("cachedir").cache()
def symLegendreBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    legendrefunc = 1/((2**basis_idx)*math.factorial(basis_idx)) * sympy.diff((z**2 - 1)**basis_idx,z,basis_idx)
    return legendrefunc
