"""
This module contains some utility function related to cobrapy community models.
"""
import pandas as pd
import cobra
import libsbml
import os
import re


def make_string_sbml_id_compatible(string, remove_ascii_escapes=False, remove_trailing_underscore=False):
    """
    This function
    :param string:
    :return:
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
    ascii_pattern = re.compile("__\d+__")
    while re.search(ascii_pattern, text):
        text = re.sub(ascii_pattern, remove_dunder_from_ascii_escape, text)
    return text


def remove_dunder_from_ascii_escape(match_obj):
    if match_obj.group() is not None:
        print(match_obj.group())
        return match_obj.group()[1:-1]


def read_medium_from_file(file, comp="_exchg"):
    medium_df = pd.read_csv(file, sep=",")
    medium_dict = {}
    for idx, row in medium_df.iterrows():
        met = row["compounds"]
        flux = float(row["maxFlux"])
        rxn = "EX_" + met + comp
        medium_dict[rxn] = flux
    return medium_dict


def load_named_model(file, format="sbml"):
    name = os.path.split(file)[1].replace(str(os.path.splitext(file)[1]), "")
    if format == "sbml":
        model = cobra.io.read_sbml_model(file)
    elif format == "json":
        model = cobra.io.load_json_model(file)
    elif format == "mat":
        model = cobra.io.load_matlab_model(file)
    else:
        raise ValueError(f"Incorrect format for model. Please use either sbml or json.")
    return model, name


def load_named_models_from_dir(path, format="sbml"):
    endings = {"sbml": ".xml", "json": ".json", "mat": ".mat"}
    named_models = {}
    files = os.listdir(path)
    expected_ending = endings[format]
    for file in files:
        if not os.path.isfile(os.path.join(path, file)) or expected_ending not in file:
            continue
        else:
            model, name = load_named_model(os.path.join(path, file), format=format)
            named_models[name] = model
    return named_models


def close_to_zero(num, t=10**-10):
    return -t < num < t


def get_model_biomass_compound(model, expected_biomass_id="", generate_if_none=False):
    """This will produce a biomass metabolite with a unique production reaction"""
    objective = str(model.objective.expression).split("*")[1].split(' ')[0]
    biomass_rxn = model.reactions.get_by_id(objective)
    biomass_products = model.reactions.get_by_id(objective).products
    biomass_met = None
    if len(expected_biomass_id) > 0:
        if expected_biomass_id in [met.id for met in biomass_products]:
            biomass_met = model.metabolites.get_by_id(expected_biomass_id)
        elif expected_biomass_id in [met.id for met in model.metabolites]:
            print(f"WARNING: expected biomass id {expected_biomass_id} is not a product of the objective function.")
            biomass_met = model.metabolites.get_by_id(expected_biomass_id)
        else:
            raise AssertionError(f"Expected biomass metabolite {expected_biomass_id} is not found in the model.")
    elif len(biomass_products) == 0:
        # No metabolites produced
        if generate_if_none:
            print(f"Note: no products in the objective function, adding biomass to it.")
            biomass_met = cobra.Metabolite("cpd11416_exchg", name='Biomass', compartment='exchg')
            model.add_metabolites([biomass_met])
            biomass_rxn.add_metabolites({biomass_met: 1.})
        else:
            raise AssertionError(f"No biomass compound could be found in objective\nObjective id: {objective}")
    elif len(biomass_products) == 1:
        biomass_met = biomass_products[0]
    else:
        # Multiple products in the objective, making biomass metabolites ambiguous
        if generate_if_none:
            print(f"Note: no products in the objective function, adding biomass to it.")
            biomass_met = cobra.Metabolite("cpd11416_exchg", name='Biomass', compartment='exchg')
            model.add_metabolites([biomass_met])
            biomass_rxn.add_metabolites({biomass_met: 1.})
        else:
            raise AssertionError(f"Multiple products in objective, biomass metabolite is ambiguous. Please set it "
                                 f"manually.\nObjective id: {objective}")
    return biomass_met


def make_model_ids_sbml_conform(model):
    for met in model.metabolites:
        if not met.name:
            met.name = met.id
        met.id = make_string_sbml_id_compatible(met.id, remove_ascii_escapes=True, remove_trailing_underscore=True)
        met.compartment = make_string_sbml_id_compatible(met.compartment, remove_ascii_escapes=True, remove_trailing_underscore=True)
    for rxn in model.reactions:
        if not rxn.name:
            rxn.name = rxn.id
        rxn.id = make_string_sbml_id_compatible(rxn.id, remove_ascii_escapes=True, remove_trailing_underscore=True)
    for group in model.groups:
        if not group.name:
            group.name = group.id
        group.id = make_string_sbml_id_compatible(group.id, remove_ascii_escapes=True, remove_trailing_underscore=True)

    rename_dict = {}
    for gene in model.genes:
        if not gene.name:
            gene.name = gene.id
        rename_dict[gene.id] = make_string_sbml_id_compatible(gene.id, remove_ascii_escapes=True, remove_trailing_underscore=True)

    if rename_dict:
        cobra.manipulation.modify.rename_genes(model, rename_dict)

    model.repair()

    return model


def get_metabolite_id_without_compartment(metabolite):
    compartment_string = "_" + metabolite.compartment
    if metabolite.id[-len(compartment_string):] == compartment_string:
        return metabolite.id[:-len(compartment_string)]
    else:
        return metabolite.id


def list_contains_unique_strings(str_list):
    return len(str_list) == len(list(set(str_list)))


def list_of_strings_is_self_contained(str_list):
    self_contained = False
    for idx, string in enumerate(str_list):
        for other_idx in range(len(str_list)):
            if idx == other_idx: continue
            elif string in str_list[other_idx]:
                self_contained = True
    return self_contained


def list_without_element(list_var, element):
    list_var = list_var.copy()
    list_var.remove(element)
    return list_var


def check_metabolite_equal_mass(met1, met2):
    """
    This function compares mass and charge of two metabolites. It returns True if the metabolites have equal mass and
    charge and False if they do not.
    """
    test_reaction = cobra.Reaction()
    test_reaction.add_metabolites({met1: -1., met2: 1.})
    return not bool(test_reaction.check_mass_balance())


def get_exchange_metabolites(model):
    exchange_metabolites = {}
    for reaction in model.exchanges:
        if len(reaction.metabolites) != 1:
            print(f"Error: exchange reaction {reaction.id} has more than 1 metabolite")
        exchange_met = list(reaction.metabolites.keys())[0]
        exchange_metabolites[exchange_met.id] = exchange_met
    return exchange_metabolites


def check_mass_balance_of_metabolites_with_identical_id(model_1, model_2):
    exchg_mets_1 = get_exchange_metabolites(model_1)
    exchg_mets_2 = get_exchange_metabolites(model_2)

    unbalanced_metabolites = []

    for met_id in set(exchg_mets_1) & set(exchg_mets_2):
        equal_mass = check_metabolite_equal_mass(exchg_mets_1[met_id], exchg_mets_2[met_id])
        if not equal_mass:
            unbalanced_metabolites.append(met_id)

    return unbalanced_metabolites


def create_parameter_in_sbml_model(sbml_model, pid, is_constant, value=None):
    """
    Helper function to set a parameter with ID and value to a SBML model.
    """
    parameter = sbml_model.createParameter()
    parameter.setId(pid)
    if value is not None:
        parameter.setValue(value)
    parameter.setConstant(is_constant)


def create_abundance_parameter(sbml_model, member_id, abundance=None):
    parameter_prefix = "Abundance_"
    create_parameter_in_sbml_model(sbml_model=sbml_model, pid=parameter_prefix+member_id, is_constant=False, value=abundance)


def read_sbml_model_from_file(file_path):
    sbml_doc = cobra.io.sbml._get_doc_from_filename(file_path)
    return sbml_doc.getModel()


def get_abundance_parameters_from_sbml_doc(sbml_model):
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


def get_abundance_parameters_from_sbml_file(sbml_file):
    sbml_model = read_sbml_model_from_file(sbml_file)
    return get_abundance_parameters_from_sbml_doc(sbml_model)


def find_matching_annotations(met1, met2):
    shared_annotation_keys = set(met1.annotation) & set(met2.annotation)
    matching_annotations = {}
    for key in shared_annotation_keys:
        if met1.annotation[key] == met2.annotation[key]:
            matching_annotations[key] = met1.annotation[key]

    return matching_annotations


def check_annotation_overlap_of_metabolites_with_identical_id(model_1, model_2):
    exchg_mets_1 = get_exchange_metabolites(model_1)
    exchg_mets_2 = get_exchange_metabolites(model_2)

    metabolites_without_overlap = []

    for met_id in set(exchg_mets_1) & set(exchg_mets_2):
        matching_annotations = find_matching_annotations(exchg_mets_1[met_id], exchg_mets_2[met_id])
        if not matching_annotations:
            metabolites_without_overlap.append(met_id)

    return metabolites_without_overlap
