import logging
from typing import TYPE_CHECKING

import cobra.exceptions
import numpy as np
import pandas as pd
import multiprocessing
from optlang.symbolics import Zero
import traceback
import os
import queue as _queue  # for Queue.Empty when draining status queue


import time
from .utils import get_f_reactions, find_incoherent_bounds
from .logger import configure_logger, get_logger_conf, get_logger_name, get_logger

import cobra
from cobra.core import Configuration
from pycomo.helper.spawnprocesspool import SpawnProcessPool

if TYPE_CHECKING:
    from cobra import Model

logger = logging.getLogger(get_logger_name())
logger.info('Multiprocess Logger initialized.')

configuration = Configuration()


def _init_fva_worker(model: "Model", ko_candidate_ids: list, status_queue=None, logger_conf=None) -> None:
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
    global _status_queue
    global logger
    pid = os.getpid()
    _model = model
    _status_queue = status_queue
    _f_rxn_set = set(get_f_reactions(_model))
    _exchg_rxn_set = set(_model.exchanges)
    _ll_candidates = set(_model.reactions.get_by_any(ko_candidate_ids))
    # if logger_conf is not None:
    #     configure_logger(logger_conf[1], logger_conf[2], with_name=logger_conf[0])
    #     logger = get_logger(logger_conf[0])
    if _status_queue is not None:
        _status_queue.put({"verbosity": "debug",
                          "pid": pid,
                          "status": f"Worker {pid} initialized.", 
                          "timestamp": time.time(), 
                          "target": None})
    logger.debug(f"_init_worker finished in {time.time() - s_time}")


def log_call_by_verbosity(verbosity):
    if verbosity.lower() == "info":
        return logger.info
    if verbosity.lower() == "warning":
        return logger.warning
    if verbosity.lower() == "error":
        return logger.error
    if verbosity.lower() == "debug":
        return logger.debug
    return logger.debug


def log_or_queue_message(verbosity, status, target=None):
    pid = os.getpid()
    if _status_queue is not None:
        _status_queue.put({"verbosity": verbosity, "pid": pid, "status": status, "timestamp": time.time(), "target": target})
    else:
        log_call_by_verbosity(verbosity)(status)


def _add_loopless_constraints_and_objective(fluxes):
    """
    Converts the model into a structure that allows for the removal of futile cycles for a given solution. This is
    achieved by fixing the direction of reactions as found in the solution, fixing the fluxes of exchange reactions
    and minimizing the remaining flux values. This approach is adapted from
    `CycleFreeFLux <https://doi.org/10.1093/bioinformatics/btv096>`_ and its implementation in COBRApy.

    :param fluxes: flux vector of the solution where loops should be removed
    """
    log_or_queue_message(verbosity="debug", status="Starting add loopless constraints", target=None)

    _model.objective = _model.solver.interface.Objective(
        Zero, direction="min", sloppy=True
    )

    objective_vars = []

    # Fix exchange reactions
    log_or_queue_message(verbosity="debug", status="Add exchange constraints", target=None)

    ko_cand_rxn_set = _ll_candidates

    exchange_rxns = _exchg_rxn_set - _f_rxn_set
    for rxn in exchange_rxns:
        flux = fluxes[rxn.id]
        rxn.bounds = (flux, flux)

    # Fix ko_candidate reactions
    log_or_queue_message(verbosity="debug", status="Add ko_candidate constraints", target=None)
    ko_candidate_rxns = ko_cand_rxn_set - _f_rxn_set - _exchg_rxn_set
    for rxn in ko_candidate_rxns:
        flux = fluxes[rxn.id]
        if flux > 0:
            lower_bound = max(0., rxn.lower_bound)
            upper_bound = min(flux, rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is lower than rxn.lower_bound due to numerical problems
                log_or_queue_message(verbosity="warning", 
                                     status=f"Flux of reaction {rxn.id} is lower than lower bound: {flux} < {lower_bound}", 
                                     target=None)
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
            objective_vars.append(rxn.forward_variable)
        elif flux < 0:
            lower_bound = max(flux, rxn.lower_bound)
            upper_bound = min(0., rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is higher than rxn.upper_bound due to numerical problems
                log_or_queue_message(
                    verbosity="warning", 
                    status=f"Flux of reaction {rxn.id} is higher than upper bound: {flux} > {upper_bound}", 
                    target=None)
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
            objective_vars.append(rxn.reverse_variable)
        else:
            rxn.bounds = 0, 0

    # Fix remaining reactions
    log_or_queue_message(verbosity="debug", status="Add remaining constraints", target=None)
    remaining_rxns = set(_model.reactions) - ko_cand_rxn_set - _f_rxn_set - _exchg_rxn_set
    for rxn in remaining_rxns:
        flux = fluxes[rxn.id]
        if flux > 0:
            lower_bound = max(0., rxn.lower_bound)
            upper_bound = min(flux, rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is lower than rxn.lower_bound due to numerical problems
                log_or_queue_message(
                    verbosity="warning", 
                    status=f"Flux of reaction {rxn.id} is lower than lower bound: {flux} < {lower_bound}", 
                    target=None)
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
        elif flux < 0:
            lower_bound = max(flux, rxn.lower_bound)
            upper_bound = min(0., rxn.upper_bound)
            if not (upper_bound >= lower_bound):
                # flux is higher than rxn.upper_bound due to numerical problems
                log_or_queue_message(
                    verbosity="warning", 
                    status=f"Flux of reaction {rxn.id} is higher than upper bound: {flux} > {upper_bound}", 
                    target=None)
                upper_bound = lower_bound
            rxn.bounds = lower_bound, upper_bound
        else:
            rxn.bounds = 0, 0

    log_or_queue_message(verbosity="debug", status="add loopless constraints finished", target=None)
    _model.objective.set_linear_coefficients({v: 1.0 for v in objective_vars})
    log_or_queue_message(verbosity="debug", status="add loopless constraints finished", target=None)
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
    log_or_queue_message(verbosity="debug", status=f"Starting loopless FVA step for rxn {rxn_id}", target=rxn_id)

    try:
        rxn = _model.reactions.get_by_id(rxn_id)
        perform_loopless_on_rxn = True
        if rxn not in _ll_candidates:
            perform_loopless_on_rxn = False
        elif rxn.boundary:
            perform_loopless_on_rxn = False

        if perform_loopless_on_rxn:
            log_or_queue_message(verbosity="info", status=f"Loop correction will be applied on {rxn.id}", target=rxn_id)


        _model.objective = rxn.id
        solution = _model.optimize("minimize")
        log_or_queue_message(verbosity="debug", status=f"{rxn.id} solver status on min is {solution.status}", target=rxn_id)

        if not solution.status == "infeasible":
            if perform_loopless_on_rxn:
                log_or_queue_message(verbosity="debug", status=f"{rxn.id} Starting loop correction", target=rxn_id)
                with _model:
                    _add_loopless_constraints_and_objective(solution.fluxes)
                    log_or_queue_message(verbosity="debug", status=f"{rxn.id} Optimize for loopless flux", target=rxn_id)
                    solution = _model.optimize()
                    log_or_queue_message(verbosity="debug", status=f"{rxn.id} Optimization for loopless flux finished", target=rxn_id)
            min_flux = solution.fluxes[rxn.id] if not solution.status == "infeasible" else 0.
        else:
            log_or_queue_message(verbosity="warning", status=f"{rxn.id} min flux is infeasible", target=rxn_id)
            min_flux = float("nan")
        solution = _model.optimize("maximize")
        log_or_queue_message(verbosity="debug", status=f"{rxn.id} solver status on max is {solution.status}", target=rxn_id)
        if not solution.status == "infeasible":
            if perform_loopless_on_rxn:
                log_or_queue_message(verbosity="debug", status=f"{rxn.id} Starting loop correction", target=rxn_id)
                with _model:
                    _add_loopless_constraints_and_objective(solution.fluxes)
                    log_or_queue_message(verbosity="debug", status=f"{rxn.id} Optimize for loopless flux", target=rxn_id)
                    solution = _model.optimize()
                    log_or_queue_message(verbosity="debug", status=f"{rxn.id} Optimization for loopless flux finished", target=rxn_id)
            max_flux = solution.fluxes[rxn.id] if not solution.status == "infeasible" else 0.
        else:
            log_or_queue_message(verbosity="warning", status=f"{rxn.id} max flux is infeasible", target=rxn_id)
            max_flux = float("nan")
        log_or_queue_message(verbosity="debug", status=f"loopless FVA step for rxn {rxn_id} finished with min/max flux {min_flux}/{max_flux}", target=rxn_id)
        return rxn_id, max_flux, min_flux
    except Exception as e:
        log_or_queue_message(verbosity="debug", status=f"Error thrown in FVA step {rxn_id}", target=rxn_id)
        return f"Error: {e}\n{traceback.format_exc()}"


def loopless_fva(pycomo_model,
                 reactions,
                 fraction_of_optimum=None,
                 use_loop_reactions_for_ko=False,
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
        log_call = logger.info
    else:
        log_call = logger.debug
    
    log_call("Starting loopless FVA")

    log_call("Preparing model")

    reaction_ids = [r.id for r in reactions]

    with pycomo_model.model:  # Revert changes to the model after fva
        log_call("Model prepared")
        if use_loop_reactions_for_ko:
            log_call("Searching for reactions that are part of loops")
            ko_candidate_ids = list(pycomo_model.get_loops(reactions=reaction_ids, processes=processes)["reaction"])
            log_call(f"Search complete. {len(ko_candidate_ids)} reactions found in loops. Proceeding with FVA.")
        elif ko_candidate_ids is None:
            ko_candidate_ids = [r.id for r in pycomo_model.model.reactions]

        if fraction_of_optimum is not None:  # Set the fraction of optimum as constraints
            log_call(f"Setting the fraction of the optimum to {fraction_of_optimum * 100}%")
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

        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        status_queue = manager.Queue()  

        try:
            if processes > 1:
                chunk_size = len(reaction_ids) // processes
                chunk_size = 1
                time_out_step = min(100, int((max_time_out-time_out)/2.))
                if time_out_step < 1: time_out_step = 1
                failed_tasks = []
                with SpawnProcessPool(
                        processes,
                        initializer=_init_fva_worker,
                        initargs=(pycomo_model.model, ko_candidate_ids, status_queue, get_logger_conf()),
                ) as pool:
                    statuses = {}
                    async_results = [pool.apply_async(_loopless_fva_step, args=(r,)) for r in reaction_ids]
                    for input_rxn, res in zip(reaction_ids, async_results):
                        # drain status_queue (non-blocking) to update local statuses immediately
                        try:
                            while True:
                                msg = status_queue.get_nowait()
                                statuses[msg["pid"]] = msg
                                if msg["verbosity"] in ["info", "debug", "warning", "error"]:
                                    if msg["verbosity"] == "debug":
                                        logger.debug(f"worker {msg['pid']}: {msg['status']}")
                                    if msg["verbosity"] == "info":
                                        logger.info(f"worker {msg['pid']}: {msg['status']}")
                                    if msg["verbosity"] == "warning":
                                        logger.warning(f"worker {msg['pid']}: {msg['status']}")
                                    if msg["verbosity"] == "error":
                                        logger.error(f"worker {msg['pid']}: {msg['status']}")
                        except _queue.Empty:
                            pass
                        try:
                            res_tuple = res.get(timeout=time_out)
                            if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                                logger.error(f"Worker error captured:\n{res_tuple}")
                                raise ValueError(f"Worker error captured:\n{res_tuple}")
                            rxn_id, max_flux, min_flux = res_tuple
                            processed_rxns += 1
                            if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                            result.at[rxn_id, "maximum"] = max_flux
                            result.at[rxn_id, "minimum"] = min_flux
                            #for worker in pool._pool._pool:
                                #logger.debug(f"Worker {worker.pid} is alive: {worker.is_alive()}")
                        except multiprocessing.TimeoutError:
                            logger.warning(f"FVA step timed out for rxn {input_rxn}")
                            failed_tasks.append(input_rxn)
                    if failed_tasks:
                        for worker in pool._pool._pool:
                            if not worker.is_alive():
                                logger.warning(f"Worker {worker.pid} is dead: {not worker.is_alive()}")
                            logger.debug(f"Worker {worker.pid} is alive: {worker.is_alive()}")
                        time_out += time_out_step
                        time_out = min(max_time_out, time_out)
                        logger.info(f"Repeating failed FVA steps for reactions: {failed_tasks}")
                        while failed_tasks and time_out <= max_time_out:
                            logger.debug(f"Repeat FVA steps with timeout = {time_out}: {failed_tasks}")
                            repeat_failed_tasks = []
                            async_results = [pool.apply_async(_loopless_fva_step, args=(r,)) for r in failed_tasks]
                            for input_rxn, res in zip(failed_tasks, async_results):
                                try:
                                    res_tuple = res.get(timeout=time_out)
                                    rxn_id, max_flux, min_flux = res_tuple
                                    processed_rxns += 1
                                    if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                        logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                                    result.at[rxn_id, "maximum"] = max_flux
                                    result.at[rxn_id, "minimum"] = min_flux
                                except multiprocessing.TimeoutError:
                                    logger.warning(f"FVA step timed out again for rxn {input_rxn}")
                                    repeat_failed_tasks.append(input_rxn)
                            for worker in pool._pool._pool:
                                if not worker.is_alive():
                                    logger.warning(f"Worker {worker.pid} is dead: {not worker.is_alive()}")
                                logger.debug(f"Worker {worker.pid} is alive: {worker.is_alive()}")
                            failed_tasks = repeat_failed_tasks
                            time_out += time_out_step
                        if failed_tasks:
                            logger.error(f"FVA failed for several reactions:\n{failed_tasks}")
                            # Single core fallback
                            logger.info(f"Running single core FVA fallback for reactions {failed_tasks}")
                            _init_fva_worker(pycomo_model.model, ko_candidate_ids)
                            for res_tuple in map(_loopless_fva_step, failed_tasks):
                                if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                                    logger.error(f"Worker error captured:\n{res_tuple}")
                                    raise ValueError(f"Worker error captured:\n{res_tuple}")
                                rxn_id, max_flux, min_flux = res_tuple
                                processed_rxns += 1
                                if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                    logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                                result.at[rxn_id, "maximum"] = max_flux
                                result.at[rxn_id, "minimum"] = min_flux
            else:
                _init_fva_worker(pycomo_model.model, ko_candidate_ids)
                for res_tuple in map(_loopless_fva_step, reaction_ids):
                    if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                        logger.error(f"Worker error captured:\n{res_tuple}")
                        raise ValueError(f"Worker error captured:\n{res_tuple}")
                    rxn_id, max_flux, min_flux = res_tuple
                    processed_rxns += 1
                    if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                        logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                    result.at[rxn_id, "maximum"] = max_flux
                    result.at[rxn_id, "minimum"] = min_flux
        finally:
            try:
                for p in multiprocessing.active_children():
                    try:
                        p.terminate()
                        p.join(timeout=0.1)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Stopping active child processes raised an exception: {e}")


            logger.debug(f"Shutting down queue manager")
            try:
                # Manager.shutdown() will stop the manager process and its queues
                manager.shutdown()
            except Exception as e:
                #logger.warning(f"Manager raised exception during shutdown: {e}")
                pass

    return result


def _fva_step(rxn_id):
    """
    Performs a single step in FVA. The
    output of the step is the minimum and maximum flux for the given reaction ID.

    :param rxn_id: The target reaction
    :return rxn_id, max_flux, min_flux: The input reaction ID, maximum flux and minimum flux (as calculated by loopless
    FVA)
    """
    log_or_queue_message(verbosity="debug", status=f"Starting FVA step for rxn {rxn_id}", target=rxn_id)
    try:
        rxn = _model.reactions.get_by_id(rxn_id)

        _model.objective = rxn.id
        log_or_queue_message(verbosity="debug", status=f"Running minimize for rxn {rxn_id}", target=rxn_id)
        solution = _model.optimize("minimize")
        log_or_queue_message(verbosity="debug", status=f"Running minimize finished for rxn {rxn_id}", target=rxn_id)
        log_or_queue_message(verbosity="debug", status=f"{rxn.id} solver status on min is {solution.status}", target=rxn_id)
        if not solution.status == "infeasible":
            min_flux = solution.fluxes[rxn.id] if not solution.status == "infeasible" else 0.
        else:
            log_or_queue_message(verbosity="warning", status=f"{rxn.id} min flux is infeasible", target=rxn_id)
            min_flux = float("nan")
        log_or_queue_message(verbosity="debug", status=f"Running maximize for rxn {rxn_id}", target=rxn_id)
        solution = _model.optimize("maximize")
        log_or_queue_message(verbosity="debug", status=f"Running maximize finished for rxn {rxn_id}", target=rxn_id)
        log_or_queue_message(verbosity="debug", status=f"{rxn.id} solver status on max is {solution.status}", target=rxn_id)
        if not solution.status == "infeasible":
            max_flux = solution.fluxes[rxn.id] if not solution.status == "infeasible" else 0.
        else:
            log_or_queue_message(verbosity="warning", status=f"{rxn.id} max flux is infeasible", target=rxn_id)
            max_flux = float("nan")
        log_or_queue_message(verbosity="debug", status=f"FVA step for rxn {rxn_id} finished with min/max flux {min_flux}/{max_flux}", target=rxn_id)
        return rxn_id, max_flux, min_flux
    except Exception as e:
        log_or_queue_message(verbosity="error", status=f"Error thrown in FVA step {rxn_id}", target=rxn_id)
        return f"Error: {e}\n{traceback.format_exc()}"
    

def fva(pycomo_model,
        reactions,
        fraction_of_optimum=None,
        verbose=False,
        processes=None,
        time_out=30,
        max_time_out=300):
    """
    Performs flux variability analysis.

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
        log_call = logger.info
    else:
        log_call = logger.debug

    log_call("Starting FVA")
    log_call("Preparing model")

    reaction_ids = [r.id for r in reactions]

    with pycomo_model.model:  # Revert changes to the model after fva
        log_call("Model prepared")

        if fraction_of_optimum is not None:  # Set the fraction of optimum as constraints
            log_call(f"Setting the fraction of the optimum to {fraction_of_optimum * 100}%")
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

        ctx = multiprocessing.get_context("spawn")
        manager = ctx.Manager()
        status_queue = manager.Queue()  

        try:
            if processes > 1:
                chunk_size = len(reaction_ids) // processes
                chunk_size = 1
                time_out_step = min(100, int((max_time_out-time_out)/2.))
                if time_out_step < 1: time_out_step = 1
                failed_tasks = []
                with SpawnProcessPool(
                        processes,
                        initializer=_init_fva_worker,
                        initargs=(pycomo_model.model, [], status_queue, get_logger_conf()),
                ) as pool:
                    statuses = {}
                    async_results = [pool.apply_async(_fva_step, args=(r,)) for r in reaction_ids]
                    for input_rxn, res in zip(reaction_ids, async_results):
                        try:
                            while True:
                                msg = status_queue.get_nowait()
                                statuses[msg["pid"]] = msg
                                if msg["verbosity"] in ["info", "debug", "warning", "error"]:
                                    if msg["verbosity"] == "debug":
                                        logger.debug(f"worker {msg['pid']}: {msg['status']}")
                                    if msg["verbosity"] == "info":
                                        logger.info(f"worker {msg['pid']}: {msg['status']}")
                                    if msg["verbosity"] == "warning":
                                        logger.warning(f"worker {msg['pid']}: {msg['status']}")
                                    if msg["verbosity"] == "error":
                                        logger.error(f"worker {msg['pid']}: {msg['status']}")
                        except _queue.Empty:
                            pass
                        try:
                            res_tuple = res.get(timeout=time_out)
                            if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                                logger.error(f"Worker error captured:\n{res_tuple}")
                                raise ValueError(f"Worker error captured:\n{res_tuple}")
                            rxn_id, max_flux, min_flux = res_tuple
                            processed_rxns += 1
                            if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                            result.at[rxn_id, "maximum"] = max_flux
                            result.at[rxn_id, "minimum"] = min_flux
                            #for worker in pool._pool._pool:
                                #logger.debug(f"Worker {worker.pid} is alive: {worker.is_alive()}")
                        except multiprocessing.TimeoutError:
                            logger.warning(f"FVA step timed out for rxn {input_rxn}")
                            failed_tasks.append(input_rxn)
                    if failed_tasks:
                        for worker in pool._pool._pool:
                            if not worker.is_alive():
                                logger.warning(f"Worker {worker.pid} is dead: {not worker.is_alive()}")
                            logger.debug(f"Worker {worker.pid} is alive: {worker.is_alive()}")
                        time_out += time_out_step
                        time_out = min(max_time_out, time_out)
                        logger.info(f"Repeating failed FVA steps for reactions: {failed_tasks}")
                        while failed_tasks and time_out <= max_time_out:
                            repeat_failed_tasks = []
                            async_results = [pool.apply_async(_fva_step, args=(r,)) for r in failed_tasks]
                            for input_rxn, res in zip(failed_tasks, async_results):
                                try:
                                    res_tuple = res.get(timeout=time_out)
                                    rxn_id, max_flux, min_flux = res_tuple
                                    processed_rxns += 1
                                    if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                        logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                                    result.at[rxn_id, "maximum"] = max_flux
                                    result.at[rxn_id, "minimum"] = min_flux
                                except multiprocessing.TimeoutError:
                                    logger.warning(f"FVA step timed out again for rxn {input_rxn}")
                                    repeat_failed_tasks.append(input_rxn)
                            failed_tasks = repeat_failed_tasks
                            time_out += time_out_step
                        if failed_tasks:
                            logger.error(f"FVA failed for several reactions:\n{failed_tasks}")
                            # Single core fallback
                            logger.info(f"Running single core FVA fallback for reactions {failed_tasks}")
                            _init_fva_worker(pycomo_model.model, [])
                            for res_tuple in map(_fva_step, failed_tasks):
                                if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                                    logger.error(f"Worker error captured:\n{res_tuple}")
                                    raise ValueError(f"Worker error captured:\n{res_tuple}")
                                rxn_id, max_flux, min_flux = res_tuple
                                processed_rxns += 1
                                if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                    logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                                result.at[rxn_id, "maximum"] = max_flux
                                result.at[rxn_id, "minimum"] = min_flux
            else:
                _init_fva_worker(pycomo_model.model, [])
                for res_tuple in map(_fva_step, reaction_ids):
                    if isinstance(res_tuple, str) and res_tuple.startswith("Error:"):  # Identify error messages
                        logger.error(f"Worker error captured:\n{res_tuple}")
                        raise ValueError(f"Worker error captured:\n{res_tuple}")
                    rxn_id, max_flux, min_flux = res_tuple
                    processed_rxns += 1
                    if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                        logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of fva steps")
                    result.at[rxn_id, "maximum"] = max_flux
                    result.at[rxn_id, "minimum"] = min_flux
        finally:
            try:
                for p in multiprocessing.active_children():
                    try:
                        p.terminate()
                        p.join(timeout=0.1)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Stopping active child processes raised an exception: {e}")


            logger.debug(f"Shutting down queue manager")
            try:
                # Manager.shutdown() will stop the manager process and its queues
                manager.shutdown()
            except Exception as e:
                #logger.warning(f"Manager raised exception during shutdown: {e}")
                pass


    return result
