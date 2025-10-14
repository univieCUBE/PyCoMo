# This file inculdes helper functions for PyCoMo models with MILP functionalities.

from numpy import linspace
from sympy import Add
import sympy

OPTLANG_BINARY = 'binary'

def make_mu_bins(mu_lb, mu_ub, n_mu_bins):
    bounds = linspace(mu_lb, mu_ub, n_mu_bins+1)
    bins = zip(bounds[:-1], bounds[1:])
    return tuple(((x[0]+x[1])/2, x) for x in bins)

def mu_approx_resolution(mu_ub, mu_lb, n_mu_bins):
    return (mu_ub - mu_lb) / n_mu_bins

def petersen_linearization(b, x, z = None, M=1000):
    """
    PETERSEN, C,,
    "A Note on Transforming the Product of Variables to Linear Form in Linear CLIFFORD Programs,"
    Working Paper, Purdue University, 1971.

    Performs Petersen Linearization of a product
    z = b*x <=> z - b*x = 0
    <=>
    {   x + M*b - M <= z <= M*b
    {   z <= x

    where :
    * b is a binary variable
    * f a linear combination of continuous or integer variables y

    :param x:   Must be an expression or variable
    :param b:   Must be a binary optlang variable
    :param z:   Must be an optlang variable. Will be mapped to the product so
                that z = b*f(y)
    :param M:   big-M constraint
    :return:
    """
    
    assert(b.type == OPTLANG_BINARY)

    if z is None:
        name = '__MUL__'.join([b.name, x.name])
        z = sympy.Symbol(name = name)
    else:
        name = z.name

    # 1st Petersen constraint
    # x + M*b - M <= z
    # x + M*b - z <= M
    cons1 = {"name" : name + '_1',
                            "expression": x + M*b - z,
                            "lb":0,
                            "ub": M}
    # 2nd Petersen constraint
    # z <= M*b
    # 0 <= M*b - z
    cons2 = {"name": name + '_2',
                            "expression": M*b - z,
                            "lb": 0,
                            "ub":None}

    # 3rd Petersen constraint
    # z <= x
    # 0 <= x - z
    cons3 = {"name": name + '_3',
                            "expression": x - z,
                            "lb": 0,
                            "ub": None,
            }

    return z, [cons1,cons2,cons3]
