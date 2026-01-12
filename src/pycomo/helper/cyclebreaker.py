"""
This module implements breaking thermodynamically infeasible cycles (TICs, here called cycles) using EFM enumeration and MILP.
"""
import pandas as pd
import cobra
import os
import re
import numpy as np
from .spawnprocesspool import SpawnProcessPool
import multiprocessing
import queue as _queue  # for Queue.Empty when draining status queue
from cobra.core import Configuration
import time
import traceback
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from .logger import configure_logger, get_logger_conf, get_logger_name, get_logger
from .utils import get_f_reactions, find_incoherent_bounds, relax_reaction_constraints_for_zero_flux
from .efmtool_interface import run_efmtool_with_custom_model

logger = logging.getLogger(get_logger_name())
logger.debug('CycleBreaker Logger initialized.')

configuration = Configuration()

def prepare_model_for_cycle_enumeration(com_model):
    """
    Prepare a community model for cycle enumeration.
    
    :param com_model: the community model
    :returns: A prepared model (simplified and bound-free structure removed)
    """
    # Remember starting configuration
    was_fixed_growth_rate = False
    mu_c = None
    if com_model.fixed_growth_rate_flag:
        was_fixed_growth_rate = True
        logger.debug(f"Model is in fixed growth rate")
        mu_c = com_model.mu_c
        com_model.apply_fixed_growth_rate(0.)
    else:
        com_model.convert_to_fixed_growth_rate(mu_c=0.)
    
    # Remove fraction metabolites
    model = com_model.convert_to_model_without_fraction_metabolites_by_copy()

    # Revert changes to community model
    if was_fixed_growth_rate:
        com_model.apply_fixed_growth_rate(mu_c)
    else:
        com_model.convert_to_fixed_abundance()

    # Remove the medium
    model.medium = {}

    # Remove demand constraints (non-zero flux enforced)
    relax_reaction_constraints_for_zero_flux(model)

    model = remove_blocked_reactions(model)

    return model

def remove_blocked_reactions(model):
    """
    Finds and removes blocked reactions.
    
    :param model: model where blocked reactions should be removed
    :returns: model without blocked reactions
    """
    logger.debug("Removing blocked reactions")
    blocked_rxns = cobra.flux_analysis.find_blocked_reactions(model)
    logger.debug(f"{len(blocked_rxns)} blocked reactions found")
    model.remove_reactions(blocked_rxns, remove_orphans=True)
    logger.debug(f"{len(blocked_rxns)} blocked reactions removed")
    return model

def enumerate_cycles_in_com_model(com_model):
    model = prepare_model_for_cycle_enumeration(com_model)
    cycles = run_efmtool_with_custom_model(custom_model=model, ref_com_model=com_model, mu=0.)
    return cycles

def add_cycle_breaker_constraint(com_model, cycle, eps=None, no_enforce_activity_constraint=True):
    """
    Adds a sum constraint for a given cycle. The constraint specifies, that the reactions of this cycle cannot be active at the same time.
    This is implemented by binary variables for reaction activity. The constraint is set so the sum of these binary variables must be lower 
    than the number of reactions in the cycle (thus at least one is inactive).

    :param com_model: The model to add the constraint
    :param cycle: The cycle to form the constraint for. A dictionairy of reaction ids and flux values.
    :param eps: A small, greater than 0 threshold, above which the reaction is considered active (default is solver tolerance)
    """
    if not isinstance(cycle, dict):
        cycle = dict(cycle)

    bigM = 1000
    if eps is None:
        eps = com_model.model.tolerance

    cb_vars = []

    for rxn_id, val in cycle.items():
        rxn = com_model.model.reactions.get_by_id(rxn_id)
        if val > 0.:
            # binary variable
            var_name = f"cb_{rxn.id}_fwd"
            if var_name in com_model.model.variables:
                y_fwd = com_model.model.variables[var_name]
            else:
                y_fwd = com_model.model.solver.interface.Variable(var_name, type="binary")
                com_model.model.solver.add([y_fwd])
                c_fwd1 = com_model.model.solver.interface.Constraint(rxn.flux_expression - bigM * y_fwd, ub=0, name=f"{var_name}_ub")
                com_model.model.solver.add([c_fwd1])
                if not no_enforce_activity_constraint:
                    c_fwd2 = com_model.model.solver.interface.Constraint(rxn.flux_expression - eps * y_fwd, lb=0, name=f"{var_name}_lb")
                    com_model.model.solver.add([c_fwd2])
            cb_vars.append(y_fwd)
            
        elif val < 0.:
            # binary variable
            var_name = f"cb_{rxn.id}_rev"
            if var_name in com_model.model.variables:
                y_rev = com_model.model.variables[var_name]
            else:
                y_rev = com_model.model.solver.interface.Variable(var_name, type="binary")
                com_model.model.solver.add([y_rev])
                c_rev1 = com_model.model.solver.interface.Constraint(rxn.flux_expression + bigM * y_rev, lb=0, name=f"{var_name}_lb")
                com_model.model.solver.add([c_rev1])
                if not no_enforce_activity_constraint:
                    c_rev2 = com_model.model.solver.interface.Constraint(rxn.flux_expression + eps * y_rev, ub=0, name=f"{var_name}_ub")
                    com_model.model.solver.add([c_rev2])
            cb_vars.append(y_rev)

    
    # Add cycle constraint
    n = len(cb_vars)

    cycle_constraint_name = get_free_cycle_constraint_name(com_model)
        
    if int(cycle_constraint_name.split("cb_cycle_")[1]) % 1000 == 0:
        logger.debug(f"Adding constraint {cycle_constraint_name}")
    c_cycle_break = com_model.model.solver.interface.Constraint(sum(cb_vars), ub=n-1, name=cycle_constraint_name)
    
    com_model.model.solver.add([c_cycle_break])

def get_free_cycle_constraint_name(com_model):
    """
    Create a name for cycle constraints in the form of cb_cycle_[number] with [number] 
    being the lowest integer where this name is not yet in the model constraints.

    :param com_model: Community model where the constraint should be added
    :return: A name that is not yet used in the model
    """
    free_cycle_name = "cb_cycle_1"
    i = 1

    while free_cycle_name in com_model.model.solver.constraints:
        i += 1
        free_cycle_name = f"cb_cycle_{i}"
    
    return free_cycle_name

def add_cycle_breaker_constraints_for_all_cycles(com_model, cycle_df, eps=None, no_enforce_activity_constraint=True):
    """
    Adds sum constraint for all cycles in the dataframe. The constraints specify, that the reactions of a cycle cannot be active at the same time.
    This is implemented by binary variables for reaction activity. The constraint is set so the sum of these binary variables must be lower 
    than the number of reactions in the cycle (thus at least one is inactive).

    :param com_model: The model to add the constraints
    :param cycle_df: A dataframe of reaction ids (columns) and cycles (rows) with corresponding flux values as cell values
    :param eps: A small, greater than 0 threshold, above which the reaction is considered active (default is solver tolerance)
    """
    for _, row in cycle_df.iterrows():
        add_cycle_breaker_constraint(com_model=com_model, cycle=dict(row), eps=eps, no_enforce_activity_constraint=no_enforce_activity_constraint)
