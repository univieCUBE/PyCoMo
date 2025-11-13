"""
This module contains some utility function related to cobrapy community models.
"""
import pandas as pd
import cobra
import os
import re
import numpy as np
from pycomo.helper.spawnprocesspool import SpawnProcessPool
import multiprocessing
import queue as _queue  # for Queue.Empty when draining status queue
from cobra.core import Configuration
import time
import traceback
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pycomo.helper.logger import configure_logger, get_logger_conf, get_logger_name, get_logger

logger = logging.getLogger(get_logger_name())
logger.info('Utils Logger initialized.')

configuration = Configuration()


def make_string_sbml_id_compatible(string, remove_ascii_escapes=False, remove_trailing_underscore=False):
    """
    This function converts a string into SBML id compatible format. This format only allows alphanumeric characters
    a-z, A-Z, 0-9 and the underscore character _.

    :param string: The string to be reformatted
    :param remove_ascii_escapes: Remove any ascii escapes in the form of digits surrounded by two underscores
    :param remove_trailing_underscore: Remove any trailing underscore characters
    :return: The formatted string
    """
    alphanum_pattern = re.compile(r'\w')

    for idx, character in enumerate(string):
        if not alphanum_pattern.match(character):
            string = string[:idx] + "_" + string[idx+1:]

    if remove_ascii_escapes:
        string = remove_ascii_escape_from_string(string)

    if remove_trailing_underscore:
        while string and string[-1] == "_":
            string = string[:-1]

    if string[0].isdigit():
        string = "_" + string

    return string


def remove_ascii_escape_from_string(text):
    """
    Remove any ascii escapes in the form of digits surrounded by two underscores, by removing the outer underscore
    characters until no more ascii escape patterns remain.

    :param text: The string to be reformatted
    :return: The formatted string
    """
    ascii_pattern = re.compile("__\d+__")
    while re.search(ascii_pattern, text):
        text = re.sub(ascii_pattern, remove_dunder_from_ascii_escape, text)
    return text


def remove_dunder_from_ascii_escape(match_obj):
    """
    Remove any ascii escapes in the form of digits surrounded by two underscores, by removing the outer underscore
    characters

    :param match_obj: The match object containing the ascii escape
    :return: The match object with the outer underscore characters removed
    """
    if match_obj.group() is not None:
        logger.debug(match_obj.group())
        return match_obj.group()[1:-1]


def read_medium_from_file(file, comp):
    """
    The file needs to be in the following format: A csv file with two columns separated by a comma (,). The two
    columns are named compounds and maxFlux. The compounds columns contains the metabolite IDs of the boundary
    metabolites as in the community metabolic model. The maxFlux column contains the maximum influx of the
    respective value as an integer or float.

    :param file: Path to the medium file
    :param comp: The name of the shared exchange compartment (required to convert metabolite IDs into exchange reaction
        IDs)
    :return: A dictionary containing the medium with exchange reactions as keys and flux constraints as values
    """
    medium_df = pd.read_csv(file, sep=",")
    medium_dict = {}
    for idx, row in medium_df.iterrows():
        met = row["compounds"]
        flux = float(row["maxFlux"])
        rxn = "EX_" + met + comp
        medium_dict[rxn] = flux
    return medium_dict


def read_abundance_from_file(file):
    """
    Read community member abundance from file. The file needs to be a csv file, with comma (,) as separator. It needs
    to contains two columns, model and fraction. Model is the name of the community member model and fraction is the
    abundance fraction for the respective communtiy member.

    :param file: Path to the abundance csv file
    :return: An abundance dictionary with community members as keys and fractions as values
    """
    endings = {".sbml", ".xml", ".json", ".mat", ".yaml", ".yml"}
    abd_df = pd.read_csv(file, sep=",", header=None)
    abd_dict = {}
    if len(abd_df.columns) != 2:
        raise ValueError("Abundance file must contain exactly 2 columns")
    abd_df.columns = ["model", "fraction"]
    for idx, row in abd_df.iterrows():
        model = row["model"]
        if str(os.path.splitext(model)[1]) in endings:
            model = model.replace(str(os.path.splitext(model)[1]), "")
        fraction = float(row["fraction"])
        abd_dict[model] = fraction
    return abd_dict


def load_named_model(file, file_format="sbml"):
    """
    Loads a metabolic model from file and returns the model together with the model file name as name.

    :param file: Path to the model file
    :param file_format: The file format of the model. Supported formats are sbml, json, mat and yaml/yml
    :return: the metabolic model as COBRApy model and the model name
    """
    name = os.path.split(file)[1].replace(str(os.path.splitext(file)[1]), "")
    if file_format == "sbml":
        model = cobra.io.read_sbml_model(file)
    elif file_format == "json":
        model = cobra.io.load_json_model(file)
    elif file_format == "mat":
        model = cobra.io.load_matlab_model(file)
    elif file_format in ["yaml", "yml"]:
        model = cobra.io.load_yaml_model(file)
    else:
        raise ValueError(f"Incorrect format for model. Please use either sbml or json.")
    return model, name


def load_named_models_from_dir(path, file_format="sbml"):
    """
    Loads all metabolic models from file in the specified directory and of given format. Supported formats are sbml,
    json, mat and yaml/yml.

    :param path: Path to the model file directory
    :param file_format: The file format of the models. Supported formats are sbml, json, mat and yaml/yml
    :return: A dictionary with model names as keys and metabolic models (COBRApy model objects) as values
    """
    endings = {"sbml": [".xml"], "json": [".json"], "mat": [".mat"], "yaml": [".yaml", ".yml"]}
    named_models = {}
    files = os.listdir(path)
    expected_ending = endings[file_format]
    for file in files:
        if not os.path.isfile(os.path.join(path, file)) or str(os.path.splitext(file)[1]) not in expected_ending:
            continue
        else:
            model, name = load_named_model(os.path.join(path, file), file_format=file_format)
            named_models[name] = model
    return named_models


def close_to_zero(num, t=None):
    """
    Checks whether a number is within threshold t of 0. Default threshold t is the solver tolerance.

    :param num: The number to be checked
    :param t: The threshold around 0
    :return: True if the number is within threshold t of 0, otherwise False
    """
    if t is None:
        t = cobra.Configuration().tolerance
    return -t < num < t


def get_model_biomass_compound(model, shared_compartment_name, expected_biomass_id="", generate_if_none=False,
                               return_biomass_rxn=False):
    """
    Finds the biomass metabolite in a model. A biomass metabolite can also be created, if the biomass reaction
    (objective) does not produce any metabolites.

    :param model: The model to be searched for a biomass metabolite
    :param shared_compartment_name: The ID of the shared exchange compartment
    :param expected_biomass_id: The ID of the biomass metabolite if already known or suspected
    :param generate_if_none: If True, a biomass metabolite is created if the objective function does not produce a
        metabolite
    :param return_biomass_rxn: If True, return the identified biomass reaction as well
    :raises KeyError: This error is raised if the biomass metabolite cannot be determined
    :return: The biomass metabolite as COBRApy metabolite
    """
    if str(model.objective.expression) in ["0.0", "0"]:
        raise ValueError(f"Model objective is not set.")
    elif "*" not in str(model.objective.expression):
        raise ValueError(f"Unknown objective format. Biomass reaction could not be determined."
                         f"\nObjective: {str(model.objective.expression)}")

    objective = str(model.objective.expression).split("*")[1].split(' ')[0]
    biomass_rxn = model.reactions.get_by_id(objective)
    logger.info(f"Identified biomass reaction from objective: {biomass_rxn.id}")
    biomass_products = model.reactions.get_by_id(objective).products
    biomass_met = None
    if len(expected_biomass_id) > 0:
        if expected_biomass_id in [met.id for met in biomass_products]:
            biomass_met = model.metabolites.get_by_id(expected_biomass_id)
        elif expected_biomass_id in [met.id for met in model.metabolites]:
            logger.warning(f"WARNING: expected biomass id {expected_biomass_id} is not a product of the objective "
                           f"function.")
            biomass_met = model.metabolites.get_by_id(expected_biomass_id)
            biomass_producing_reactions = []
            for rxn in biomass_met.reactions:
                if biomass_met in rxn.products:
                    biomass_producing_reactions.append(rxn)
            if len(biomass_producing_reactions) == 1:
                biomass_rxn = biomass_producing_reactions[0]
                logger.info(f"Identified biomass reaction expected biomass metabolite: {biomass_rxn.id}")
            elif len(biomass_producing_reactions) == 0:
                logger.warning(f"No reaction in the model is producing the expected biomass metabolite "
                               f"{expected_biomass_id}!")
                biomass_rxn = None
            elif len(biomass_producing_reactions) > 1:
                logger.warning(f"Multiple reactions produce the expected biomass metabolite {expected_biomass_id}")
                biomass_rxn = None
        else:
            raise KeyError(f"Expected biomass metabolite {expected_biomass_id} is not found in the model.")
    elif len(biomass_products) == 0:
        # No metabolites produced
        if generate_if_none:
            logger.info(f"Note: no products in the objective function, adding biomass to it.")
            biomass_met = cobra.Metabolite(f"cpd11416_{shared_compartment_name}", name='Biomass',
                                           compartment=shared_compartment_name)
            model.add_metabolites([biomass_met])
            biomass_rxn.add_metabolites({biomass_met: 1.})
        else:
            raise KeyError(f"No biomass compound could be found in objective\nObjective id: {objective}")
    elif len(biomass_products) == 1:
        biomass_met = biomass_products[0]
    else:
        # Multiple products in the objective, making biomass metabolites ambiguous
        if generate_if_none:
            logger.info(f"Note: no products in the objective function, adding biomass to it.")
            biomass_met = cobra.Metabolite(f"cpd11416_{shared_compartment_name}", name='Biomass',
                                           compartment=shared_compartment_name)
            model.add_metabolites([biomass_met])
            biomass_rxn.add_metabolites({biomass_met: 1.})
        else:
            raise KeyError(f"Multiple products in objective, biomass metabolite is ambiguous. Please set it "
                           f"manually.\nObjective id: {objective}")
    logger.debug(f"Final identified biomass rxn: {biomass_rxn}: {biomass_rxn.metabolites}")
    if return_biomass_rxn:
        return biomass_met, biomass_rxn
    return biomass_met


def make_model_ids_sbml_conform(model):
    """
    Reformat the IDs of metabolites, reactions and genes to ensure they are SBML conform. Also removes any ASCII
    escapes in the form of digits surrounded by two underscores each.

    :param model: The model to be reformatted
    :return: The reformatted model
    """
    for met in model.metabolites:
        if not met.name:
            met.name = met.id
        met.id = make_string_sbml_id_compatible(met.id, remove_ascii_escapes=True, remove_trailing_underscore=True)
        met.compartment = make_string_sbml_id_compatible(met.compartment, remove_ascii_escapes=True,
                                                         remove_trailing_underscore=True)
    for rxn in model.reactions:
        if not rxn.name:
            rxn.name = rxn.id
        rxn.id = make_string_sbml_id_compatible(rxn.id, remove_ascii_escapes=True, remove_trailing_underscore=True)
    for group in model.groups:
        if not group.name:
            group.name = group.id
        if group.id:
            group.id = make_string_sbml_id_compatible(group.id,
                                                      remove_ascii_escapes=True,
                                                      remove_trailing_underscore=True)

    rename_dict = {}
    for gene in model.genes:
        if not gene.name:
            gene.name = gene.id
        rename_dict[gene.id] = make_string_sbml_id_compatible(gene.id, remove_ascii_escapes=True,
                                                              remove_trailing_underscore=True)

    if rename_dict:
        cobra.manipulation.modify.rename_genes(model, rename_dict)

    model.repair()

    return model


def get_metabolite_id_without_compartment(metabolite):
    """
    Get the ID of a metabolite without the compartment string, should it be included in the metabolite ID.

    :param metabolite: The metabolite whose ID should be extracted
    :return: The metabolite ID without the compartment string
    """
    compartment_string = "_" + metabolite.compartment
    if metabolite.id[-len(compartment_string):] == compartment_string:
        return metabolite.id[:-len(compartment_string)]
    else:
        return metabolite.id


def list_contains_unique_strings(str_list):
    """
    Checks if a list contains only unique strings

    :param str_list: List of strings to be checked
    :return: True, if all strings in the list are unique, else False
    """
    return len(str_list) == len(list(set(str_list)))


def list_of_strings_is_self_contained(str_list):
    """
    Checks if a list of strings contains strings that are substrings of other string within the list.

    :param str_list: The list of strings to be checked
    :return: True, if strings in the list are substrings of others, else False
    """
    self_contained = False
    for idx, string in enumerate(str_list):
        for other_idx in range(len(str_list)):
            if idx == other_idx:
                continue
            elif string in str_list[other_idx]:
                self_contained = True
    return self_contained


def list_without_element(list_var, element):
    """
    Returns a copy of a list without the specified element.

    :param list_var: List whose element should be excluded
    :param element: Element to be excluded
    :return: A list without target element
    """
    list_var = list_var.copy()
    list_var.remove(element)
    return list_var


def check_metabolite_equal_mass(met1, met2):
    """
    This function compares mass and charge of two metabolites. It returns True if the metabolites have equal mass and
    charge and False if they do not.

    :param met1: metabolite 1
    :param met2: metabolite 2
    :return: True, if the metabolites have equal mass and charge, else False
    """
    test_reaction = cobra.Reaction()
    test_reaction.add_metabolites({met1: -1., met2: 1.})
    return not bool(test_reaction.check_mass_balance())


def get_exchange_metabolites(model):
    """
    Retrieve all metabolites of a model that are part of exchange reactions.

    :param model: The model whose exchange metabolite should be extracted
    :return: Dictionary of exchange metabolite with metabolite IDs as keys and COBRApy metabolites as values
    """
    exchange_metabolites = {}
    for reaction in model.exchanges:
        if len(reaction.metabolites) != 1:
            logger.error(f"Error: exchange reaction {reaction.id} has more than 1 metabolite")
        exchange_met = list(reaction.metabolites.keys())[0]
        exchange_metabolites[exchange_met.id] = exchange_met
    return exchange_metabolites


def check_mass_balance_of_metabolites_with_identical_id(model_1, model_2):
    """
    Checks whether metabolites with identical IDs in each of the models has identical mass and charge.

    :param model_1: Model 1
    :param model_2: Model 2
    :return: List of metabolite IDs of metabolites that do not have identical mass and charge in both models
    """
    exchg_mets_1 = get_exchange_metabolites(model_1)
    exchg_mets_2 = get_exchange_metabolites(model_2)

    unbalanced_metabolites = []

    for met_id in set(exchg_mets_1) & set(exchg_mets_2):
        exchg_met1 = exchg_mets_1[met_id]
        exchg_met2 = exchg_mets_2[met_id]
        equal_mass = check_metabolite_equal_mass(exchg_met1, exchg_met2)
        if not equal_mass:
            unbalanced_metabolites.append(met_id)
    return unbalanced_metabolites


def create_parameter_in_sbml_model(sbml_model, pid, is_constant, value=None, as_name=False):
    """
    Helper function to set a parameter with ID and value to a SBML model.

    :param sbml_model: libsbml model
    :param pid: Parameter ID
    :param is_constant: Flag whether the parameter is constant
    :param value: Value of the parameter
    :param as_name: Set the value of the parameter as name of the parameter instead of value
    """
    parameter = sbml_model.createParameter()
    parameter.setId(pid)
    if value is not None:
        if as_name:
            parameter.setName(value)
        else:
            parameter.setValue(value)
    parameter.setConstant(is_constant)


def create_abundance_parameter(sbml_model, member_id, abundance=None):
    """
    Create a parameter for the abundance of a community member.

    :param sbml_model: libsbml model
    :param member_id: ID of the community member
    :param abundance: Abundance fraction of the community member
    """
    parameter_prefix = "Abundance_"
    create_parameter_in_sbml_model(sbml_model=sbml_model, pid=parameter_prefix+member_id, is_constant=False,
                                   value=abundance)


def read_sbml_model_from_file(file_path):
    """
    Reads a metabolic model from file and return it in libsbml format.

    :param file_path: Path to the metabolic model file
    :return: libsbml model
    """
    sbml_doc = cobra.io.sbml._get_doc_from_filename(file_path)
    return sbml_doc.getModel()


def get_abundance_parameters_from_sbml_doc(sbml_model):
    """
    Extract the community member abundance fractions from the libsbml model.

    :param sbml_model: libsbml model
    :return: Dictionary of community member abundances with community member names as keys and abundance fraction as
        values
    """
    parameter_prefix = "Abundance_"
    abundance_dict = {}

    for parameter in sbml_model.getListOfParameters():
        parameter_id = parameter.getId()
        if parameter_prefix in parameter_id[:len(parameter_prefix)]:
            fraction = None
            if parameter.isSetValue():
                fraction = parameter.getValue()
            abundance_dict[parameter_id[len(parameter_prefix):]] = fraction

    return abundance_dict


def get_flags_and_muc_from_sbml_file(sbml_file):
    """
    Retrieve the community growth rate and the flags for fixed abundance state, fixed growth state as well as the ID of
    the shared exchange compartment from a sbml file.

    :param sbml_file: Path to the SBML model file
    :return: Dictionary of flags and parameters as keys and their values as values
    """
    sbml_model = read_sbml_model_from_file(sbml_file)

    parameter_dict = {}

    for parameter in sbml_model.getListOfParameters():
        parameter_id = parameter.getId()
        if parameter_id == "mu_c":
            value = 1.
            if parameter.isSetValue():
                value = parameter.getValue()
            parameter_dict["mu_c"] = value
        if parameter_id == "fixed_abundance_flag":
            value = 0
            if parameter.isSetValue():
                value = parameter.getValue()
            value = value == 1  # Flags are stored as 1 and 0 for True and False. None is converted to False
            parameter_dict["fixed_abundance_flag"] = value
        if parameter_id == "shared_compartment_id":
            if not parameter.isSetName():
                raise ValueError("Error: Missing parameter shared_compartment_id (parameter name should contain ID of "
                                 "the shared compartment)")
            name = parameter.getName()
            parameter_dict["shared_compartment_name"] = name
        if parameter_id == "fixed_growth_rate_flag":
            value = 0
            if parameter.isSetValue():
                value = parameter.getValue()
            value = value == 1  # Flags are stored as 1 and 0 for True and False. None is converted to False
            parameter_dict["fixed_growth_rate_flag"] = value

    return parameter_dict


def get_abundance_parameters_from_sbml_file(sbml_file):
    """
    Retrieve the community member abundance fractions from a sbml file.

    :param sbml_file: Path to the SBML model file
    :return: Dictionary of community member abundances with community member names as keys and abundance fraction as
        values
    """
    sbml_model = read_sbml_model_from_file(sbml_file)
    return get_abundance_parameters_from_sbml_doc(sbml_model)


def find_matching_annotations(met1, met2):
    """
    Finds keys of the metabolite annotation dictionary that are present in both metabolites and whose values are
    identical

    :param met1: Metabolite 1
    :param met2: Metabolite 2
    :return: A dictionary of identical annotations with annotation keys as keys and their values as values
    """
    shared_annotation_keys = set(met1.annotation) & set(met2.annotation)
    matching_annotations = {}
    for key in shared_annotation_keys:
        if met1.annotation[key] == met2.annotation[key]:
            matching_annotations[key] = met1.annotation[key]

    return matching_annotations


def check_annotation_overlap_of_metabolites_with_identical_id(model_1, model_2):
    """
    Retrieves all metabolites with identical IDs in both models, that do not share a single overlap in any of their
    metabolite annotation key / value pairs.

    :param model_1: Model 1
    :param model_2: Model 2
    :return: Metabolites with identical IDs in both models, but no overlap in their annotation
    """
    exchg_mets_1 = get_exchange_metabolites(model_1)
    exchg_mets_2 = get_exchange_metabolites(model_2)

    metabolites_without_overlap = []

    for met_id in set(exchg_mets_1) & set(exchg_mets_2):
        matching_annotations = find_matching_annotations(exchg_mets_1[met_id], exchg_mets_2[met_id])
        if not matching_annotations:
            metabolites_without_overlap.append(met_id)

    return metabolites_without_overlap

def get_metabolites_without_elements_from_model(model):
    """
    Retrieves all metabolites without elements in a model. This is important for assessing mass balance - 
    without elements in metabolites, mass balance cannot be calculated.

    :param model: The model to check
    :return: A list of metabolites without elements
    """
    metabolites_without_elements = []
    for met in model.metabolites:
        if not check_element_presence_in_metabolite(met):
            metabolites_without_elements.append(met)
    return metabolites_without_elements


def check_element_presence_in_metabolite(met):
    """
    Checks if elements are present in a metabolite. This is important for assessing mass balance - 
    without elements in metabolites, mass balance cannot be calculated.

    :param met: The metabolite to check
    :return: True if elements are present, otherwise False
    """
    return len(met.elements) > 0


def check_mass_balance_fomula_safe(model):
    """
    Checks a model's reactions for mass and charge balance. In case the mass / charge balance check fails due to
    wrongly formatted metabolite formulae, add reaction and metabolites to unbalanced reactions and throw a warning.

    :param model: The model to check
    :return: Dictionary of key reaction and value: dictionary (metabolite: float)
    """
    _NOT_MASS_BALANCED_TERMS = {
        "SBO:0000627",  # EXCHANGE
        "SBO:0000628",  # DEMAND
        "SBO:0000629",  # BIOMASS
        "SBO:0000631",  # PSEUDOREACTION
        "SBO:0000632",  # SINK
    }

    unbalanced = {}
    for reaction in model.reactions:
        if reaction.annotation.get("sbo") not in _NOT_MASS_BALANCED_TERMS:
            try:
                balance = reaction.check_mass_balance()
                if balance:
                    unbalanced[reaction] = balance
            except ValueError:
                logger.warning(f"Warning: not wrongly formatted metabolite formula in metabolites of reaction {reaction.id}")
                unbalanced[reaction] = reaction.metabolites
    return unbalanced


def get_f_reactions(model):
    """
    Get the IDs of all fraction reactionsm in a PyCoMo community metabolic model.

    :param model: A PyCoMo community metabolic model
    :return: A set of all fraction reactions
    """
    return model.reactions.query(
        lambda x: (x.id[:3] == "SK_" and x.id[-3:] in {"_lb", "_ub"}) or "_fraction_reaction" in x.id)


def relax_reaction_constraints_for_zero_flux(model):
    """
    This function relaxes all constraints of a model to allow a flux of 0 in all reactions.

    :param model: Model to be changed
    """
    for reaction in model.reactions:
        if reaction.lower_bound > 0.:
            reaction.lower_bound = 0.
        if reaction.upper_bound < 0.:
            reaction.upper_bound = 0.


def _init_loop_worker(model, status_queue=None):
    """
    Initialize a global model object for multiprocessing and attach a QueueHandler
    so that worker logging goes into the provided multiprocessing queue.

    :param model: The model to perform find loops in
    :param log_queue: multiprocessing.Queue used to transport LogRecords to main process
    """

    global _model
    global _status_queue
    global logger
    _model = model
    _status_queue = status_queue
    pid = os.getpid()

    if _status_queue is not None:
        _status_queue.put({"verbosity": "debug",
                          "pid": pid,
                          "status": f"Worker {pid} initialized.", 
                          "timestamp": time.time(), 
                          "target": None})
    else:
        logger.debug(f"Worker {pid} initialized.")

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


def _find_loop_step(rxn_id, check_feasibility=False):
    try:
        pid = os.getpid()
        log_or_queue_message(verbosity="debug", status=f"Starting find_loop_step for {rxn_id}", target=rxn_id)
        rxn = _model.reactions.get_by_id(rxn_id)
        _model.objective = rxn.id
        t = 10 * cobra.Configuration().tolerance 
        log_or_queue_message(verbosity="status", status=f"Minimize {rxn_id}", target=rxn_id)

        if check_feasibility:
            with _model as temp_model:
                rxn = temp_model.reactions.get_by_id(rxn_id)
                if rxn.lower_bound > -t:
                    min_flux = 0.
                else:
                    rxn.upper_bound = -t
                    temp_model.objective = 0
                    temp_model.slim_optimize()
                    if temp_model.solver.status == "infeasible":
                        min_flux = 0.
                    elif temp_model.solver.status == "optimal":
                        min_flux = -1.
                    else:
                        log_or_queue_message(verbosity="warning", status=f"{pid}: Finished minimize {rxn_id} with status {solution.status}", target=rxn_id)
                        min_flux = float("nan")            
        else:
            solution = _model.optimize("minimize")
            #logger.debug(f"{pid}: Finished minimize {rxn_id} with status {solution.status}")
            if solution.status != "optimal":
                log_or_queue_message(verbosity="warning", status=f"{pid}: Finished minimize {rxn_id} with status {solution.status}", target=rxn_id)
            min_flux = solution.objective_value if not solution.status == "infeasible" else float("nan")

        log_or_queue_message(verbosity="status", status=f"Maximize {rxn_id}", target=rxn_id)
        
        if check_feasibility:
            with _model as temp_model:
                rxn = temp_model.reactions.get_by_id(rxn_id)
                if rxn.upper_bound < t:
                    max_flux = 0.
                else:
                    rxn.lower_bound = t
                    temp_model.objective = 0
                    temp_model.slim_optimize()
                    if temp_model.solver.status == "infeasible":
                        max_flux = 0.
                    elif temp_model.solver.status == "optimal":
                        max_flux = 1.
                    else:
                        max_flux = float("nan")
                        log_or_queue_message(verbosity="warning", status=f"{pid}: Finished maximize {rxn_id} with status {solution.status}", target=rxn_id)
                             
        else:
            solution = _model.optimize("maximize")
            #logger.debug(f"{pid}: Finished minimize {rxn_id} with status {solution.status}")
            if solution.status != "optimal":
                log_or_queue_message(verbosity="warning", status=f"{pid}: Finished maximize {rxn_id} with status {solution.status}", target=rxn_id)
            max_flux = solution.objective_value if not solution.status == "infeasible" else float("nan")

        log_or_queue_message(verbosity="status", status=f"Finished {rxn_id}", target=rxn_id)

        return rxn_id, max_flux, min_flux
    except Exception as e:
        log_or_queue_message(verbosity="error", status=f"Error at {rxn_id}", target=rxn_id)
        return f"{pid}: Error: {e}\n{traceback.format_exc()}"


def find_loops_in_model(model, reactions=None, processes=None, time_out=300, max_time_out=None, restart_on_stall=False):
    """
    This function finds thermodynamically infeasible cycles in models. This is accomplished by setting the medium to
    contain nothing and relax all constraints to allow a flux of 0. Then, FVA is run on all reactions.

    :param model: Model to be searched for thermodynamically infeasible cycles
    :param processes: The number of processes to use
    :return: A dataframe of reactions and their flux range, if they can carry non-zero flux without metabolite input
    """
    loop_model = model.copy()
    loop_model.medium = {}
    relax_reaction_constraints_for_zero_flux(loop_model)
    loops = []
    if reactions is None:
        reaction_ids = [r.id for r in loop_model.reactions]
    else:
        reaction_ids = [r if isinstance(r, str) else r.id for r in reactions]

    max_time_out = max(time_out * 2, len(loop_model.reactions)*0.5)

    num_rxns = len(reaction_ids)

    if processes is None:
        processes = configuration.processes

    processes = min(processes, num_rxns)
    processed_rxns = 0

    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    status_queue = manager.Queue()       

    try:
        if processes > 1:
            chunk_size = len(reaction_ids) // processes
            time_out_step = min(100, int((max_time_out-time_out)/2.))
            failed_tasks = []
            #manager = multiprocessing.Manager()
            #status_dict = manager.dict()
            
            pool = SpawnProcessPool(
                processes,
                initializer=_init_loop_worker,
                initargs=(tuple([loop_model, status_queue])),
            )
            try:
                async_results = [pool.apply_async(_find_loop_step, args=(r,)) for r in reaction_ids]
                pending = list(zip(reaction_ids, async_results))
                last_processed = processed_rxns

                rounds_since_last_result = 0
                # local dict to hold latest status per worker pid
                statuses = {}
                last_restart_at = 0

                while pending:
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

                    found_result = False
                    # if len(statuses) == 0:
                    #     logger.debug("No workers active - trying again")
                    #     time.sleep(1)
                    #     continue
                    for i, (input_rxn, res) in enumerate(pending):
                        try:
                            res_tuple = res.get(timeout=0.00001)  # Short timeout
                            rxn_id, max_flux, min_flux = res_tuple
                            processed_rxns += 1
                            rounds_since_last_result = 0
                            if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of find loops steps")
                                for pid, info in statuses.items():
                                    logger.debug(f"Worker {pid}: {info}")
                            if min_flux != 0. or max_flux != 0.:
                                loops.append({"reaction": rxn_id, "min_flux": min_flux, "max_flux": max_flux})
                            pending.pop(i)
                            found_result = True
                            break  # Only process one result per loop
                        except multiprocessing.TimeoutError:
                            # logger.debug(f"Timeout waiting for result of rxn {input_rxn}")
                            # for pid, info in status_dict.items():
                            #     logger.debug(f"Worker {pid}: {info}")
                            continue
                    if not found_result:
                        rounds_since_last_result += 1
                        # No new result, display status queue
                        logger.debug(f"No new results, current worker status: {statuses}")
                        if len(statuses) == 0:
                            logger.debug("No workers active - trying again")
                        else:
                            # check for stalled workers
                            stalled_pids = [pid for pid, info in statuses.items() if time.time() - info["timestamp"] > time_out]
                            if stalled_pids:
                                logger.warning(f"Detected stalled worker(s): {stalled_pids} (no update for >{time_out}s)")
                                if restart_on_stall:
                                    logger.info("Restarting worker pool due to stalled workers.")
                                    # terminate current pool and recreate it
                                    try:
                                        pool.terminate()
                                    except Exception as exc:
                                        logger.warning(f"Exception when terminating pool: {exc}")
                                    try:
                                        pool.join()
                                    except Exception:
                                        logger.warning(f"Exception when joining pool: {exc}")
                                    # rebuild pool and re-submit pending tasks
                                    try:
                                        pool = SpawnProcessPool(
                                            processes,
                                            initializer=_init_loop_worker,
                                            initargs=(tuple([loop_model, status_queue])),
                                        )
                                        async_results = [pool.apply_async(_find_loop_step, args=(rxn,)) for rxn, _ in pending]
                                        pending = list(zip([rxn for rxn, _ in pending], async_results))
                                        statuses.clear()
                                        rounds_since_last_result = 0
                                        # continue loop with fresh pool

                                        time_out += time_out_step
                                        time_out = min(max_time_out, time_out)
                                        continue
                                    except Exception as exc:
                                        logger.error(f"Failed to recreate worker pool: {exc}")
                                        raise exc
                                    
                        time.sleep(min(2*rounds_since_last_result, time_out))  # Wait before next check
                # for input_rxn, res in zip(reaction_ids, async_results):
                #     try:
                #         res_tuple = res.get(timeout=time_out)
                #         rxn_id, max_flux, min_flux = res_tuple
                #         processed_rxns += 1
                #         if processed_rxns % 10 == 0 or processed_rxns == len(reaction_ids):
                #             logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of find loops steps")
                #         if min_flux != 0. or max_flux != 0.:
                #             loops.append({"reaction": rxn_id, "min_flux": min_flux, "max_flux": max_flux})
                #     except multiprocessing.TimeoutError:
                #         logger.warning(f"Find loops step timed out for rxn {input_rxn}")
                #         failed_tasks.append(input_rxn)
                logger.info(f"While pending closed!")
                if failed_tasks:
                    time_out += time_out_step
                    time_out = min(max_time_out, time_out)
                    logger.info(f"Repeating failed find loops steps for reactions: {failed_tasks}")
                    while failed_tasks and time_out <= max_time_out:
                        repeat_failed_tasks = []
                        async_results = [pool.apply_async(_find_loop_step, args=(r,)) for r in failed_tasks]
                        for input_rxn, res in zip(failed_tasks, async_results):
                            try:
                                res_tuple = res.get(timeout=time_out)
                                rxn_id, max_flux, min_flux = res_tuple
                                processed_rxns += 1
                                if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                                    logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of find loops steps")
                                if min_flux != 0. or max_flux != 0.:
                                    loops.append({"reaction": rxn_id, "min_flux": min_flux, "max_flux": max_flux})
                            except multiprocessing.TimeoutError:
                                logger.warning(f"FVA step timed out again for rxn {input_rxn}")
                                repeat_failed_tasks.append(input_rxn)
                        failed_tasks = repeat_failed_tasks
                        time_out += time_out_step
                    logger.error(f"Find loops failed for several reactions:\n{failed_tasks}")
                    # Single core fallback
                    _init_loop_worker(loop_model)
                    logger.info(f"Running single core FVA fallback for reactions {failed_tasks}")
                    for rxn_id, max_flux, min_flux in map(_find_loop_step, reaction_ids):
                        processed_rxns += 1
                        if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                            logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of find loops steps")
                        if min_flux != 0. or max_flux != 0.:
                            loops.append({"reaction": rxn_id, "min_flux": min_flux, "max_flux": max_flux})
            finally:
                # ensure pool is cleaned up
                try:
                    pool.terminate()
                except Exception as exc:
                    logger.warning(f"Exception when terminating pool: {exc}")
                try:
                    pool.join()
                except Exception:
                    logger.warning(f"Exception when joining pool: {exc}")
        else:
            logger.info(f"Running find loops in single core mode")
            _init_loop_worker(loop_model)
            for rxn_id, max_flux, min_flux in map(_find_loop_step, reaction_ids):
                processed_rxns += 1
                if processed_rxns % 100 == 0 or processed_rxns == len(reaction_ids):
                    logger.info(f"Processed {round((float(processed_rxns) / num_rxns) * 100, 2)}% of find loops steps")
                if min_flux != 0. or max_flux != 0.:
                    loops.append({"reaction": rxn_id, "min_flux": min_flux, "max_flux": max_flux})
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


    loops_df = pd.DataFrame(loops, columns=["reaction", "min_flux", "max_flux"])
    logger.info("Loop detection finished")
    return loops_df


def replace_metabolite_stoichiometry(rxn, new_stoich):
    """
    This reaction allows for safely and reversibly assigning new reaction stoichiometry.
    Assigning stoichiometry in add/subtract_metabolites method of cobrapy with combine set to False results in a
    KeyError, if the metabolite was not previously part of the reaction. As with the cobrapy add/subtract_metabolites
    method, the stoichiometry of metabolites that are not part of the new stoichiometry, are left unchanged.

    Example: r1: a -> b; new_stoich: {c: -1}; => r1: a + c -> b
    Example: r1: a -> b; new_stoich: {a: 0}; => r1: -> b

    :param rxn: Target reaction
    :param new_stoich: Dictionary with metabolites (string ID or cobra.Metabolite) as keys and floats as values.
    """
    stoich_to_subtract = {}
    for k, v in rxn.metabolites.items():
        if k.id in new_stoich.keys() or k in new_stoich.keys():
            stoich_to_subtract[k] = v
    rxn.subtract_metabolites(stoich_to_subtract, combine=True)
    rxn.add_metabolites(new_stoich, combine=True)
    return

def find_incoherent_bounds(model):
    """
    Find incoherent bounds (lb > ub, bounds with NaN) and return them.
    :param model: a cobrapy model
    :return: A list of reactions with incoherent bounds. If none are found, returns an empty list
    """
    result = []
    for rxn in model.reactions:
        faulty_reaction = False
        if np.isnan(rxn.lower_bound):
            logger.info(f"Lower bound of reaction {rxn.id} is NaN!")
            faulty_reaction = True
        if np.isnan(rxn.upper_bound):
            logger.info(f"Upper bound of reaction {rxn.id} is NaN!")
            faulty_reaction = True
        if rxn.lower_bound > rxn.upper_bound:
            logger.info(f"Lower bound exceeds upper bound in reaction {rxn.id}!")
            faulty_reaction = True
        if faulty_reaction:
            result.append(rxn)
    return result
