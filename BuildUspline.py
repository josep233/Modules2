import sys
import Uspline

if __name__ == "CubitPythonInterpreter_2":
    # We are running within the Coreform Cubit application, cubit python module is already available
    pass
else:
    sys.path.append("C:\Program Files\Coreform Cubit 2022.4\bin")
    import cubit
    cubit.init([])

################################### 
######## B-Spline examples ########
###################################
def example_1():  
  # Test-1
  spline_space = { "domain": [0, 2], "degree": [2, 2], "continuity": [-1, 1, -1]}
  Uspline.make_uspline_mesh( spline_space, "two_element_quadratic_bspline" )

def example_2():  
  # Test-2
  spline_space = { "domain": [0, 3], "degree": [2, 2, 2], "continuity": [-1, 1, 1, -1]}
  Uspline.make_uspline_mesh( spline_space, "three_element_quadratic_bspline" )

def example_3():  
  # Test-3
  spline_space = { "domain": [0, 3], "degree": [2, 2, 2], "continuity": [-1, 2, 2, -1]}
  Uspline.make_uspline_mesh( spline_space, "supersmooth_quadratic_bspline" )

def example_4():  
  # Test-4
  degree = 6
  continuity = [-1]
  for i in range( degree ):
    continuity.append( degree - 1 )
  continuity.append( -1 )
  spline_space = { "domain": [0, 10], "degree": [degree]*(degree+1), "continuity": continuity}
  Uspline.make_uspline_mesh( spline_space, "high_order_bspline" )

################################### 
######## U-Spline examples ########
###################################
def example_5():
  # Test-5
  spline_space = { "domain": [0, 4], "degree": [1, 2, 3, 4], "continuity": [-1, 1, 2, 3, -1]}
  Uspline.make_uspline_mesh( spline_space, "multi_deg_uspline" )

def example_6():  
  # Test-6
  spline_space = { "domain": [0, 4], "degree": [1, 2, 3, 4], "continuity": [-1, 1, 2, 3, -1]}
  Uspline.make_uspline_mesh( spline_space, "multi_deg_maxsmooth_uspline" )

def example_7():
  # Test-7
  spline_space = { "domain": [0, 5], "degree": [1, 2, 3, 2, 1], "continuity": [-1, 0, 1, 1, 0, -1]}
  Uspline.make_uspline_mesh( spline_space, "ref_int_uspline" )

def example_8():  
  # Test-8
  spline_space = { "domain": [0, 11], "degree": [1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1], "continuity": [-1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, -1]}
  Uspline.make_uspline_mesh( spline_space, "optimal_multi_deg_uspline" )

example_1()