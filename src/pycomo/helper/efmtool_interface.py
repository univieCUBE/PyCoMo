"""
This module provides an interface for efmtool
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
logger.debug('EfmtoolInteface Logger initialized.')

configuration = Configuration()

def run_efmtool(matrix, rxn_names, met_names, revs=[]):
    """
    run efmtool and return numpy array with fluxes
    Args:
        matrix: numpy array with the stoichiometric matrix
        rxn_names: list of reaction names
        met_names: list of metabolite names
        revs: list of reversibilities (0 irreversible / 1 reversible). 
            If not provided, all reactions are assumed to be irreversible.
    Returns:
        pandas data frame with EFMs as rows and reactions as columns
    """
    try:
        import efmtool
    except ImportError:
        raise ImportError("Optional dependency 'efmtool' is required for EFM enumeration. "
                          "Install it with pip install pycomo[efm]")
    options = efmtool.get_default_options()
    options["arithmetic"] = "fractional"
    options['level'] = 'WARNING'

    # if reversibilities not provided, assume all reactions irreversible
    if not revs:
        revs = reversibilities = [0]*matrix.shape[1]
    modes = efmtool.calculate_efms(stoichiometry = matrix,
                                   reversibilities = revs,
                                   reaction_names = rxn_names,
                                   metabolite_names = met_names,
                                   options = options)

    nice_modes = make_nice_results(modes, rxn_names)
    
    return nice_modes

def make_nice_results(modes, rxn_names):
    """
    Convert array from efmtool into a nice pandas DataFrame
    Args:
        modes: numpy array with the output of "efmtool.calculate_efms()"
        rxn_names: list of reaction names
    Returns:
        pandas data frame with EFMs as rows and reactions as columns
    """

    # if there are no columns - no modes
    if modes.shape[1] == 0:
        return pd.DataFrame()

    # solve "ValueError: Big-endian buffer not supported on little-endian compiler"
    # modes = modes.byteswap().newbyteorder()

    res = pd.DataFrame(modes.T,
                       index = [f"Mode{(i+1)}" for i in range(modes.shape[1])],
                       columns = rxn_names)

    # normalize values by the column C
    if "C" in res.columns:
        res = res.div(res.C, axis = 0)

    return res

def community_matrix_sbml(cobra_model, organism_ids, mu=1):
    """
    Extracts stoichiometric matrix, and lists of reaction IDs, metabolite IDs 
    and reversibilities from a cobra model. Adds bounds on growth rate and the
    total sum of microbe fractions=1

    Args:
        cobra_model: cobrapy Model object created with PyCoMo
        organism_ids: list of organism names used in the cobra model
        mu: growth rate to set
    Returns:
        N: numpy array with the stoichiometric matrix formatted for the use with efmtool
        rxns: list of reaction names
        mets: list of metabolite names
        revs: list of reversibilities (0 irreversible / 1 reversible)
        """
    N = cobra.util.array.create_stoichiometric_matrix(cobra_model)
    rxns = [r.id for r in cobra_model.reactions]
    mets = [m.id for m in cobra_model.metabolites]
    revs = []
    for index, r in enumerate(cobra_model.reactions):
        if r.lower_bound < 0:
            if r.reversibility:
                revs.append(1)
            # flip irreversible reactions that only go in reverse direction
            if not r.reversibility:
                N[:, index] = -N[:, index]
                revs.append(0)
        else:
            revs.append(0)

    #N, rxns, mets, revs = add_bounds_to_pycomo_matrix(N, rxns, mets, revs, mu, organism_ids)

    return (N, rxns, mets, revs)

def project_cycles_to_non_f_rxns_no_pycomo(df):
    cols = df.columns
    new_cols = []
    for c in cols:
        if ("SK_" == c[:3] and ("_lb" == c[-3:] or "_ub" == c[-3:])):
            continue
        elif c == "f_final":
            continue
        elif "_fraction_reaction" in c:
            continue
        else:
            new_cols.append(c)
    return df[new_cols]

def project_cycles_to_non_f_rxns(df, com_model):
    f_reactions = com_model.f_reactions
    cols = df.columns
    new_cols = []
    for c in cols:
        if c in f_reactions:
            continue
        elif c == "f_final":
            continue
        else:
            new_cols.append(c)
    return df[new_cols]

def remove_zero_and_duplicate_cycles(df):
    return df.loc[~(df==0).all(axis=1)].drop_duplicates()


def run_efmtool_with_custom_model(custom_model, ref_com_model, mu=0.):
    try:
        custom_model.reactions.remove("abundance_reaction")  # remove reaction for fixing composition
    except ValueError:
        logger.debug("No abundance reaction found")
    # IDs of abundance reactions and transport reactions
    organism_ids = ref_com_model.get_member_names()
    ecfms = run_efmtool(*community_matrix_sbml(custom_model, organism_ids, mu))
    return ecfms

