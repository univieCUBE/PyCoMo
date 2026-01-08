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
