# first line: 21
@joblib.Memory("cachedir").cache()
def evalBernsteinBasis1D(variate,degree,basis_idx,domain_out):
    domain_in = [-1,1]
    z = sympy.Symbol('z')
    B = sympy.functions.combinatorial.factorials.binomial(degree, basis_idx) * (z**basis_idx) * (1 - z)**(degree - basis_idx)
    variate = COB.affineMapping(variate,domain_in,domain_out)
    ans = B.subs(z,variate) 
    return ans
