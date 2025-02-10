import logging
from typing import TYPE_CHECKING

import cobra.exceptions
import numpy as np
import pandas as pd
import multiprocessing
from optlang.symbolics import Zero
import traceback

import time
from .utils import get_f_reactions, find_incoherent_bounds
from .logger import configure_logger, get_logger_conf

import cobra
from cobra.core import Configuration
from pycomo.helper.spawnprocesspool import SpawnProcessPool

if TYPE_CHECKING:
    from cobra import Model

logger = logging.getLogger("pycomo")
logger.info('Multiprocess Logger initialized.')

configuration = Configuration()


def _init_fva_worker(model: "Model", ko_candidate_ids: list, logger_conf) -> None:
    """
    Initialize a global model object and corresponding variables for multiprocessing.

    :param model: The model to perform FVA on
    :param ko_candidate_ids: The list of candidates for loop removal
    """
    s_time = time.time()
    global _model
    global _f_rxn_set
    global _exchg_rxn_set
    global _ll_candidates
    _model = model
    _f_rxn_set = set(get_f_reactions(_model))
    _exchg_rxn_set = set(_model.exchanges)
    _ll_candidates = set(_model.reactions.get_by_any(ko_candidate_ids))
    configure_logger(logger_conf[0], logger_conf[1])
    logger.debug(f"_init_worker finished in {time.time() - s_time}")


def _add_loopless_constraints_and_objective(fluxes):
    """
    Converts the model into a structure that allows for the removal of futile cycles for a given solution. This is
    achieved by fixing the direction of reactions as found in the solution, fixing the fluxes of exchange reactions
    and minimizing the remaining flux values. This approach is adapted from
    `CycleFreeFLux <https://doi.org/10.1093/bioinformatics/btv096>`_ and its implementation in COBRApy.

    :param fluxes: flux vector of the solution where loops should be removed
    """
    logger.debug("Starting add loopless constraints")

    _model.objective = _model.solver.interface.Objective(
        Zero, direction="min", sloppy=True
    )

    objective_vars = []

    # Fix exchange reactions
    logger.debug("Add exchange constraints")

    ko_cand_rxn_set = _ll_candidates

    exchange_rxns = _exchg_rxn_set - _f_rxn_set
    for rxn in exchange_rxns:
        flux = fluxes[rxn.id]
        rxn.bounds = (flux, flux)

    # Fix ko_candidate reactions
    logger.debug("Add ko_candidate constraints")
    ko_candidate_rxns = ko_cand_rxn_set - _f_rxn_set - _exchg_rxn_set
    for rxn in ko_candidate_rxns:
        flux = fluxes[rxn.id]
        if flux > 0:
            lower_bound = max(0., rxn.lower_bound)
            upper_bound = min(flux, rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is lower than rxn.lower_bound due to numerical problems
                logger.warning(f"Flux of reaction {rxn.id} is lower than lower bound: {flux} < {lower_bound}")
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
            objective_vars.append(rxn.forward_variable)
        elif flux < 0:
            lower_bound = max(flux, rxn.lower_bound)
            upper_bound = min(0., rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is higher than rxn.upper_bound due to numerical problems
                logger.warning(f"Flux of reaction {rxn.id} is higher than upper bound: {flux} > {upper_bound}")
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
            objective_vars.append(rxn.reverse_variable)
        else:
            rxn.bounds = 0, 0

    # Fix remaining reactions
    logger.debug("Add remaining constraints")
    remaining_rxns = set(_model.reactions) - ko_cand_rxn_set - _f_rxn_set - _exchg_rxn_set
    for rxn in remaining_rxns:
        flux = fluxes[rxn.id]
        if flux > 0:
            lower_bound = max(0., rxn.lower_bound)
            upper_bound = min(flux, rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is lower than rxn.lower_bound due to numerical problems
                logger.warning(f"Flux of reaction {rxn.id} is lower than lower bound: {flux} < {lower_bound}")
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
        elif flux < 0:
            lower_bound = max(flux, rxn.lower_bound)
            upper_bound = min(0., rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is higher than rxn.upper_bound due to numerical problems
                logger.warning(f"Flux of reaction {rxn.id} is higher than upper bound: {flux} > {upper_bound}")
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
        else:
            rxn.bounds = 0, 0

    logger.debug("Set coefficients")
    _model.objective.set_linear_coefficients({v: 1.0 for v in objective_vars})
    logger.debug("add loopless constraints finished")
    return


def _loopless_fva_step(rxn_id):
    """
    Performs a single step in loopless FVA. A normal FVA step will be applied if the reaction is boundary, or not a
    reaction within loops (not a loopless candidate). Loop correction is done on the remaining candidate reactions. The
    output of the step is the minimum and maximum flux for the given reaction ID.

    :param rxn_id: The target reaction
    :return rxn_id, max_flux, min_flux: The input reaction ID, maximum flux and minimum flux (as calculated by loopless
    FVA)
    """
    logger.debug(f"Starting loopless FVA step for rxn {rxn_id}")
    try:
        rxn = _model.reactions.get_by_id(rxn_id)
        perform_loopless_on_rxn = True
        if rxn not in _ll_candidates:
            perform_loopless_on_rxn = False
        elif rxn.boundary:
            perform_loopless_on_rxn = False

        if perform_loopless_on_rxn:
            logger.info(f"Loop correction will be applied on {rxn.id}")

        _model.objective = rxn.id
        solution = _model.optimize("minimize")
        if not solution.status == "infeasible":
            if perform_loopless_on_rxn:
                logger.debug(f"{rxn.id} Starting loop correction")
                with _model:
                    _add_loopless_constraints_and_objective(solution.fluxes)
                    logger.debug(f"{rxn.id} Optimize for loopless flux")
                    solution = _model.optimize()
                    logger.debug(f"{rxn.id} Optimization for loopless flux finished")
            min_flux = solution.fluxes[rxn.id] if not solution.status == "infeasible" else 0.
        else:
            logger.debug(f"{rxn.id} min flux is infeasible")
            min_flux = 0.
        solution = _model.optimize("maximize")
        if not solution.status == "infeasible":
            if perform_loopless_on_rxn:
                logger.debug(f"{rxn.id} Starting loop correction")
                with _model:
                    _add_loopless_constraints_and_objective(solution.fluxes)
                    logger.debug(f"{rxn.id} Optimize for loopless flux")
                    solution = _model.optimize()
                    logger.debug(f"{rxn.id} Optimization for loopless flux finished")
            max_flux = solution.fluxes[rxn.id] if not solution.status == "infeasible" else 0.
        else:
            logger.debug(f"{rxn.id} max flux is infeasible")
            max_flux = 0.
        logger.debug(f"loopless FVA step for rxn {rxn_id} finished with min/max flux {min_flux}/{max_flux}")
        return rxn_id, max_flux, min_flux
    except Exception as e:
        logger.error(f"Error thrown in FVA step {rxn_id}")
        return f"Error: {e}\n{traceback.format_exc()}"


def loopless_fva(pycomo_model,
                 reactions,
                 fraction_of_optimum=None,
                 use_loop_reactions_for_ko=True,
                 ko_candidate_ids=None,
                 verbose=False,
                 processes=None,
                 time_out=30,
                 max_time_out=300):
    """
    Performs flux variability analysis and removes futile cycles from the solutions. This is
    achieved by fixing the direction of reactions as found in the solution, fixing the fluxes of exchange reactions
    and minimizing the remaining flux values. This approach is adapted from
    `CycleFreeFLux <https://doi.org/10.1093/bioinformatics/btv096>`_ and its implementation in COBRApy.

    :param pycomo_model: A pycomo community metabolic model
    :param reactions: A list of reactions that should be analysed
    :param fraction_of_optimum: The fraction of the optimal objective flux that needs to be reached
    :param use_loop_reactions_for_ko: Find loops in the model and use these reactions as ko_candidates. Overwrites
    value in ko_candidates
    :param ko_candidate_ids: Reactions to be constrained and used in the objective (as set of reaction ids)
    :param verbose: Prints progress messages
    :param processes: The number of processes to use for the calculation
    :return: A dataframe of reaction flux solution ranges. Contains the columns minimum and maximum with index of
    reaction IDs
    """

    if verbose:
        logger.info("Starting loopless FVA")
    else:
        logger.debug("Starting loopless FVA")
    if verbose:
        logger.info("Preparing model")
    else:
        logger.debug("Preparing model")

    reaction_ids = [r.id for r in reactions]

    with pycomo_model.model:  # Revert changes to the model after fva
        if verbose:
            logger.info("Model prepared")
        else:
            logger.debug("Model prepared")
        if use_loop_reactions_for_ko:
            if verbose:
                logger.info("Searching for reactions that are part of loops")
            else:
                logger.debug("Searching for reactions that are part of loops")
            ko_candidate_ids = list(pycomo_model.get_loops(processes=processes)["reaction"])
            if verbose:
                logger.info(f"Search complete. {len(ko_candidate_ids)} reactions found in loops. Proceeding with FVA.")
            else:
                logger.debug(f"Search complete. {len(ko_candidate_ids)} reactions found in loops. Proceeding with FVA.")
        elif ko_candidate_ids is None:
            ko_candidate_ids = [r.id for r in pycomo_model.model.reactions]

        if fraction_of_optimum is not None:  # Set the fraction of optimum as constraints
            if verbose:
                logger.info(f"Setting the fraction of the optimum to {fraction_of_optimum * 100}%")
            else:
                logger.debug(f"Setting the fraction of the optimum to {fraction_of_optimum * 100}%")
            fraction_of_optimum = float(fraction_of_optimum)
            if not (0. <= fraction_of_optimum <= 1.):
                logger.warning(f"fraction_of_optimum is either not numerical or outside the range of 0 - 1.\n"
                               f"Continuing with fraction_of_optimum=1")
                fraction_of_optimum = 1.0

            objective_value = pycomo_model.model.slim_optimize()
            if np.isnan(objective_value):
                raise cobra.exceptions.Infeasible("Error: Infeasible!")
            logger.debug(f"Objective Value is {objective_value}, type: {type(objective_value)}, resulting product: {fraction_of_optimum * objective_value}")
            if pycomo_model.model.solver.objective.direction == "max":
                original_objective = pycomo_model.model.problem.Variable(
                    "original_objective",
                    lb=fraction_of_optimum * objective_value,
                )
            else:
                original_objective = pycomo_model.model.problem.Variable(
                    "original_objective",
                    ub=fraction_of_optimum * objective_value,
                )
            original_objective_constraint = pycomo_model.model.problem.Constraint(
                pycomo_model.model.solver.objective.expression - original_objective,
                lb=0,
                ub=0,
                name="original_objective_constraint",
            )
            pycomo_model.model.add_cons_vars([original_objective, original_objective_constraint])

        # Carry out fva
        num_rxns = len(reaction_ids)

        result = pd.DataFrame(
            {
                "minimum": np.zeros(num_rxns, dtype=float),
                "maximum": np.zeros(num_rxns, dtype=float),
            },
            index=reaction_ids,
        )

        if processes is None:
            processes = configuration.processes

        processes = min(processes, num_rxns)

        logger.debug(f"Running with {processes} processes")

        processed_rxns = 0

        if processes > 1:
            chunk_size = len(reaction_ids) // processes
            chunk_size = 1
            time_out_step = min(100, int((max_time_out-time_out)/2.))
            if time_out_step < 1: time_out_step = 1
            failed_tasks = []
            with SpawnProcessPool(
                    processes,
                    initializer=_init_fva_worker,
                    initargs=(pycomo_model.model, ko_candidate_ids, get_logger_conf()),
            ) as pool:
                async_results = [pool.apply_async(_loopless_fva_step, args=(r,)) for r in reaction_ids]
                for input_rxn, res in zip(reaction_ids, async_results):
                    try:
                        res_tuple = res.get(timeout=time_out)
                        if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                            logger.error(f"Worker error captured:\n{res_tuple}")
                            raise ValueError(f"Worker error captured:\n{res_tuple}")
                        rxn_id, max_flux, min_flux = res_tuple
                        processed_rxns += 1
                        if processed_rxns % 10 == 0:
                            logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                        result.at[rxn_id, "maximum"] = max_flux
                        result.at[rxn_id, "minimum"] = min_flux
                        #for worker in pool._pool._pool:
                            #logger.debug(f"Worker {worker.pid} is alive: {worker.is_alive()}")
                    except multiprocessing.TimeoutError:
                        logger.warning(f"FVA step timed out for rxn {input_rxn}")
                        failed_tasks.append(input_rxn)
                if failed_tasks:
                    time_out += time_out_step
                    time_out = min(max_time_out, time_out)
                    logger.info(f"Repeating failed FVA steps for reactions: {failed_tasks}")
                    while failed_tasks and time_out <= max_time_out:
                        repeat_failed_tasks = []
                        async_results = [pool.apply_async(_loopless_fva_step, args=(r,)) for r in failed_tasks]
                        for input_rxn, res in zip(failed_tasks, async_results):
                            try:
                                res_tuple = res.get(timeout=time_out)
                                rxn_id, max_flux, min_flux = res_tuple
                                processed_rxns += 1
                                if processed_rxns % 10 == 0:
                                    logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                                result.at[rxn_id, "maximum"] = max_flux
                                result.at[rxn_id, "minimum"] = min_flux
                            except multiprocessing.TimeoutError:
                                logger.warning(f"FVA step timed out again for rxn {input_rxn}")
                                repeat_failed_tasks.append(input_rxn)
                        failed_tasks = repeat_failed_tasks
                        time_out += time_out_step
                    logger.error(f"FVA failed for several reactions:\n{failed_tasks}")
        else:
            _init_fva_worker(pycomo_model.model, ko_candidate_ids, get_logger_conf())
            for res_tuple in map(_loopless_fva_step, reaction_ids):
                if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                    logger.error(f"Worker error captured:\n{res_tuple}")
                    raise ValueError(f"Worker error captured:\n{res_tuple}")
                rxn_id, max_flux, min_flux = res_tuple
                processed_rxns += 1
                if processed_rxns % 10 == 0:
                    logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                result.at[rxn_id, "maximum"] = max_flux
                result.at[rxn_id, "minimum"] = min_flux

    return result
