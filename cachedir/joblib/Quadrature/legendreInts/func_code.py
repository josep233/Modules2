# first line: 55
@joblib.Memory("cachedir").cache()
def legendreInts(n,domain):
    M = numpy.zeros(2*n, dtype = "double")
    M[0] = 2.0
    return M
