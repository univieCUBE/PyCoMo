"""
This module implements discretized growth for optimization across the full solution space. 
With this, linearizing the model with fixed abundance or growth rate is no longer necessary. 
However, this approach ustilizes binary variable, using a MILP to solve. 
This is faster than non-linear solvers, but slower than LPs.

The implementation is based on an idea by Salvy and Hatzimanikatis:
Salvy, P., Hatzimanikatis, V. 
The ETFL formulation allows multi-omics integration in thermodynamics-compliant metabolism and expression models. 
Nat Commun 11, 30 (2020). 
https://doi.org/10.1038/s41467-019-13818-7
"""

import numpy as np
from cobra.core import Configuration
import logging
import sympy
from sympy import Add
from .logger import configure_logger, get_logger_conf, get_logger_name, get_logger
from .utils import replace_metabolite_stoichiometry, get_f_reactions, find_incoherent_bounds, relax_reaction_constraints_for_zero_flux

logger = logging.getLogger(get_logger_name())
logger.debug('Discretized Growth Logger initialized.')

configuration = Configuration()

OPTLANG_BINARY = 'binary'

def relax_model_linearisation(com_model):
    com_model.equal_abundance()
    member_names = com_model.get_member_names()

    # Relax fraction reaction
    for m in member_names:
        fraction_rxn = com_model.model.reactions.get_by_id(f"{m}_fraction_reaction")
        try:
            fraction_met = com_model.model.metabolites.get_by_id(f"{m}_f_biomass_met")
        except KeyError:
            fraction_met = com_model._backup_metabolites[f"{m}_f_biomass_met"]
            com_model.model.add_metabolites(fraction_met)
        fraction_rxn.bounds = (0., com_model.max_flux)
        replace_metabolite_stoichiometry(fraction_rxn, {fraction_met: 0.})
        bio2com_rxn = com_model.model.reactions.get_by_id(f"{m}_to_community_biomass")
        replace_metabolite_stoichiometry(bio2com_rxn, {fraction_met: 0.})
    # Remove equal growth (at fixed abundance)
    com_model.model.reactions.abundance_reaction.bounds = 0., 0.

    com_model.fixed_abundance_flag = False
    com_model.fixed_growth_rate_flag = False
    return


def init_f_variables(com_model):
    f_vars = {}
    for member in com_model.get_member_names():
        f_var = com_model.model.solver.interface.Variable(f"{member}_fraction_variable")
        f_vars[member] = f_var
        cons_f = com_model.model.solver.interface.Constraint(
            com_model.model.reactions.get_by_id(f"{member}_fraction_reaction").flux_expression - 1. * f_var, 
            lb=0., 
            ub=0., 
            name=f"{member}_fraction_constraint"
        )
        com_model.model.solver.add([f_var, cons_f])
    return f_vars


def init_mu_variables(model, r_bio, mu_lb, mu_ub, n_mu_bins):
    """
    approximate growth with binary variables.

    :return:
    """

    k_vars = list()
    N = n_mu_bins
    n_vars = int(np.ceil(np.log2(N+1)))

    for e in range(n_vars):
        k_vars.append(
            model.solver.interface.Variable(
                kind="k",
                hook=model,
                id_=str(2 ** e),
                name=str(2 ** e), 
                type="binary"
                )
                )

    the_integer = Add(*[(2 ** i) * k_i for i, k_i in enumerate(k_vars)])
    binarized_mu = mu_lb + the_integer * mu_approx_resolution(mu_ub, mu_lb, n_mu_bins)

    growth_coupling_expr = r_bio.flux_expression - binarized_mu


    cons_bin = model.problem.Constraint(
        expression=growth_coupling_expr, 
        name="k_c_bin",
        ub=0.,
        lb=0.
        )
    
    cons_mu_bound = model.problem.Constraint(
        expression=r_bio.flux_expression, 
        name="k_c_mu_bound",
        ub=mu_ub,
        lb=mu_lb
        )

    model.solver.add([cons_bin, cons_mu_bound])

    return k_vars


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
    :param M:   bigM constraint
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
    cons1 = {
        "name" : name + '_1',
        "expression": x + M*b - z,
        "lb":0,
        "ub": M
        }
    
    # 2nd Petersen constraint
    # z <= M*b
    # 0 <= M*b - z
    cons2 = {
        "name": name + '_2', 
        "expression": M*b - z,
        "lb": 0,
        "ub":None
        }

    # 3rd Petersen constraint
    # z <= x
    # 0 <= x - z
    cons3 = {
        "name": name + '_3', 
        "expression": x - z,
        "lb": 0,
        "ub": None,
        }

    return z, [cons1,cons2,cons3]


def get_ordered_k_vars(k_vars):
    """
    Returns in order the variables that discretize growth
    :return:
    """
    # k_i is a binary variable for the binary expansion f the fraction on N
    # of the max growth rate
    k_vars = sorted(k_vars, key=lambda x: int(x.name))
    return k_vars


def discretize_growth(model, r_bio, f_var, k_vars, mu_ub, mu_lb, n_mu_bins, queue=False, constraint_limits=0.):
    """
    Performs Petersen linearization on μ*G to keep a MILP problem

    :return:
    """
    G = f_var

    big_m = 1.

    k_vars = get_ordered_k_vars(k_vars)

    out_expr = - r_bio.flux_expression

    for i, ga_i in enumerate(k_vars):
        # Linearization step for k_i * G
        z_name = '__MUL__'.join([ga_i.name, G.name])
        # Add the variables
        model_z_i = model.solver.interface.Variable(
                                      name=z_name,
                                      lb=0,
                                      ub=big_m,
                                      queue=False)

        z_i, new_constraints = petersen_linearization(b=ga_i, x=G, M=big_m,
                                                      z=model_z_i)

        # Add the constraints:
        for cons in new_constraints:
            model.solver.add([model.problem.Constraint(expression=cons["expression"], 
                                                       name=cons["name"],
                                                       ub=cons["ub"],
                                                       lb=cons["lb"])])

        out_expr += (2 ** i) * mu_approx_resolution(mu_ub, mu_lb, n_mu_bins) * model_z_i

    out_expr += mu_lb * G

    cons = model.problem.Constraint(
        expression=out_expr, 
        name='__GC__'.join(["sum_c", G.name]),
        ub=constraint_limits,
        lb=-constraint_limits
        )

    model.solver.add([cons])

    return out_expr


def discretize_growth_of_model(com_model, f_vars, mu_lb=0., mu_ub=1000., n_mu_bins=10, constraint_limits=0.):
    biomass_rxn = com_model.model.reactions.community_biomass
    member_names = com_model.get_member_names()
    k_vars = init_mu_variables(com_model.model, biomass_rxn, mu_lb=mu_lb, mu_ub=mu_ub, n_mu_bins=n_mu_bins)
    for m in member_names: 
        discretize_growth(
            model=com_model.model, 
            r_bio=com_model.model.reactions.get_by_id(f"{m}_to_community_biomass"), 
            f_var=f_vars[m], 
            k_vars=k_vars,
            mu_lb=mu_lb, 
            mu_ub=mu_ub, 
            n_mu_bins=n_mu_bins, 
            queue=False,
            constraint_limits=constraint_limits)
        
