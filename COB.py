#imports needed libraries
import os
import sympy as sy
import matplotlib.pyplot as plt
import numpy
import unittest

def affineMapping(x,old_domain,new_domain):
    b1 = numpy.array(new_domain)
    b2 = numpy.array(old_domain)
    constant = (b1[-1] - b1[0]) / (b2[-1] - b2[0])
    b2tild = b2 * constant
    shift = b2tild[0] - b1[0]
    param_coord = x*constant - shift
    return param_coord
def affineMapping2(x,old_domain,new_domain):
    b1 = numpy.array(new_domain)
    b2 = numpy.array(old_domain)
    constant = (b1[-1] - b1[0]) / (b2[-1] - b2[0])
    b2tild = b2 * constant
    shift = b2tild[0] - b1[0]
    param_coord = x*constant - shift
    return param_coord, constant