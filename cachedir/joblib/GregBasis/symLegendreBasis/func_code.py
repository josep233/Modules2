# first line: 97
@joblib.Memory("cachedir").cache()
def symLegendreBasis( degree, basis_idx, domain, variate ):
    x = sympy.symbols( 'x', real = True )
    if degree == 0:
        p = sympy.Poly( 1, x )
    else:
        term_1 = 1.0 / ( ( 2.0 ** degree ) * sympy.factorial( degree ) )
        term_2 = ( ( x**2) - 1.0 ) ** degree 
        term_3 = sympy.diff( term_2, x, degree )
        p = term_1 * term_3
        p = sympy.poly( sympy.simplify( p ) )
    return p
