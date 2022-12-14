import sys
import numpy

if __name__ == "CubitPythonInterpreter_2":
    # We are running within the Coreform Cubit application, cubit python module is already available
    pass
else:
    sys.path.append(r'C:\\Program Files\\Coreform Cubit 2022.11\\bin\\')
    import cubit
    cubit.init([])

## MAIN FUNCTION
def make_uspline_mesh( spline_space, filename ):
  cubit.cmd( "reset" )
  build_geometry( spline_space )
  generate_bezier_mesh( spline_space )
  assign_uspline_params( spline_space )
  build_uspline()
  export_uspline( filename )

## SECONDARY FUNCTIONS
def build_geometry( spline_space ):
  cubit.cmd( f"create curve location {spline_space['domain'][0]} 0 0 location {spline_space['domain'][1]} 0 0" )

def generate_bezier_mesh( spline_space ):
  cubit.cmd( f"curve 1 interval {len(spline_space['degree'])}" )
  cubit.cmd( "mesh curve 1" )

def assign_uspline_params( spline_space ):
  cubit.cmd( "set uspline curve 1 degree 1 continuity 0" )
  ien_array = get_ien_array()
  elem_id_list = tuple( ien_array.keys() )
  for eidx in range( 0, len( ien_array ) ):
    eid = elem_id_list[eidx]
    elem_degree = spline_space["degree"][eidx]
    cubit.cmd( f"set uspline edge {eid} degree {elem_degree}" )
    if eidx < ( len( ien_array ) - 1 ):
      interface_nid = ien_array[eid][1]
      interface_cont = spline_space["continuity"][eidx + 1]
      cubit.cmd( f"set uspline node {interface_nid} continuity {interface_cont}" )

def build_uspline():
  cubit.cmd( "build uspline curve 1 as 1" )
  cubit.cmd( "fit uspline 1" )

def export_uspline( filename ):
  cubit.cmd( f"export uspline 1 json '{filename}'" )

## UTILITY FUNCTIONS
def get_num_elems():
  ien_array = get_ien_array()
  return max( ien_array.keys() )

def get_ien_array():
  elem_list = cubit.parse_cubit_list( "edge", "in curve 1" )
  ien_array = {}
  for eid in elem_list:
    ien_array[eid] = cubit.get_connectivity( "edge", eid )
  ien_array = sort_element_nodes( ien_array )
  return ien_array

def get_elem_centers( ien_array ):
  xc = []
  for eid in ien_array:
    x0 = cubit.get_nodal_coordinates( ien_array[eid][0] )[0]
    x1 = cubit.get_nodal_coordinates( ien_array[eid][1] )[0]
    xc.append( ( x0 + x1 ) / 2.0 )
  xc = numpy.array( xc )
  return xc

def get_ordered_elem_list():
  ien_array = get_ien_array()
  sort_idx = numpy.argsort( get_elem_centers( ien_array ) )
  sorted_elems = list( ien_array.keys() )
  return sorted_elems

def sort_element_nodes( ien_array ):
  for eid in ien_array:
    x0 = cubit.get_nodal_coordinates( ien_array[eid][0] )[0]
    x1 = cubit.get_nodal_coordinates( ien_array[eid][1] )[0]
    if x1 < x0:
      ien_array[eid] = tuple( reversed( ien_array[eid] ) )
  return ien_array

#=================================================================================================================================================
#=================================================================================================================================================
#=================================================================================================================================================
#=================================================================================================================================================
#=================================================================================================================================================


################################### 
######## B-Spline examples ########
###################################
def example_1():  
  # Test-1
  spline_space = { "domain": [0, 2], "degree": [2, 2], "continuity": [-1, 1, -1]}
  make_uspline_mesh( spline_space, "two_element_quadratic_bspline" )

def example_2():  
  # Test-2
  spline_space = { "domain": [0, 3], "degree": [2, 2, 2], "continuity": [-1, 1, 1, -1]}
  make_uspline_mesh( spline_space, "three_element_quadratic_bspline" )

def example_3():  
  # Test-3
  spline_space = { "domain": [0, 3], "degree": [2, 2, 2], "continuity": [-1, 2, 2, -1]}
  make_uspline_mesh( spline_space, "supersmooth_quadratic_bspline" )

def example_4():  
  # Test-4
  degree = 6
  continuity = [-1]
  for i in range( degree ):
    continuity.append( degree - 1 )
  continuity.append( -1 )
  spline_space = { "domain": [0, 10], "degree": [degree]*(degree+1), "continuity": continuity}
  make_uspline_mesh( spline_space, "high_order_bspline" )

################################### 
######## U-Spline examples ########
###################################
def example_5():
  # Test-5
  spline_space = { "domain": [0, 4], "degree": [1, 2, 3, 4], "continuity": [-1, 1, 2, 3, -1]}
  make_uspline_mesh( spline_space, "multi_deg_uspline" )

def example_6():  
  # Test-6
  spline_space = { "domain": [0, 4], "degree": [1, 2, 3, 4], "continuity": [-1, 1, 2, 3, -1]}
  make_uspline_mesh( spline_space, "multi_deg_maxsmooth_uspline" )

def example_7():
  # Test-7
  spline_space = { "domain": [0, 5], "degree": [1, 2, 3, 2, 1], "continuity": [-1, 0, 1, 1, 0, -1]}
  make_uspline_mesh( spline_space, "ref_int_uspline" )

def example_8():  
  # Test-8
  spline_space = { "domain": [0, 11], "degree": [1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1], "continuity": [-1, 1, 2, 3, 3, 3, 3, 3, 3, 2, 1, -1]}
  make_uspline_mesh( spline_space, "optimal_multi_deg_uspline" )

example_2()