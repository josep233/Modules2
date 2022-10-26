# first line: 29
@joblib.Memory("cachedir").cache()
def symBernsteinBasis1D(variate,degree,basis_idx,space_domain):
    z = sympy.Symbol('z')
    B = sympy.functions.combinatorial.factorials.binomial(degree, basis_idx) * (z**basis_idx) * (1 - z)**(degree - basis_idx)
    return B
