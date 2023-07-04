#!/usr/bin/env python3

"""
The pycomo module contains classes for single species and community metabolic models. They extend the cobrapy classes
by metainformation required for community model generation. The community model can be used for simulation, transfer via
 the sbml format, setting abundances and generation of FBA flux vector tables.
"""

# IMPORT SECTION
import cobra
import numpy as np
import pandas as pd
import libsbml
import os
from typing import List
from utils import *
from cli import *


class SingleOrganismModel:
    """
    This class contains a single organism metabolic model and its meta information needed for community model generation
    """
    model: cobra.Model
    name: str
    _original_name: str  # Used to make a conversion table at any point
    biomass_met: cobra.Metabolite = None
    biomass_met_id: str = None
    prepared_model: cobra.Model = None
    _name_via_annotation: str = None
    _exchange_met_name_conversion: dict = {}

    def __init__(self, model, name, biomass_met_id="", name_via_annotation=None):
        self.model = model.copy()
        self._original_name = name
        self.name = make_string_sbml_id_compatible(name, remove_ascii_escapes=True)
        if name != self.name:
            print(f"Warning: model name {name} is not compliant with sbml id standards and was changed to {self.name}")
        self.biomass_met_id = biomass_met_id
        self._name_via_annotation = name_via_annotation

    def get_name_conversion(self):
        return self._original_name, self.name

    def set_name_via_annotation(self, name_via_annotation):
        self._name_via_annotation = name_via_annotation
        if self.prepared_model is not None:
            self.prepare_for_merging()

    def rename_comp_in_met_id(self, model, old_comp, new_comp, inplace=True, change_name=False,
                              remove_comp_from_name=False):
        if not inplace:
            model = model.copy()

        for met in model.metabolites.query(lambda x: x.compartment == new_comp):
            met_string = met.id
            if old_comp in met_string and old_comp == met_string[-len(old_comp):]:
                met.id = met.id.replace("_" + old_comp, "_" + new_comp)
            else:
                # Add compartment to met.id
                met.id += f"_{new_comp}"
            if remove_comp_from_name:
                if '_' + old_comp in met.name:
                    met.name = met.name.replace("_" + old_comp, "")
                elif '-' + old_comp in met.name:
                    met.name = met.name.replace("-" + old_comp, "")
            if change_name:
                if '_' + old_comp in met.name:
                    met.name = met.name.replace("_" + old_comp, "_" + new_comp)
                else:
                    met.name += f"_{new_comp}"

        if inplace:
            return
        else:
            return model

    def rename_comp_in_rxn_id(self, model, old_comp, new_comp, inplace=True, change_name=False):
        if not inplace:
            model = model.copy()

        for rxn in model.reactions:
            rxn_string = rxn.id
            if old_comp in rxn_string and old_comp == rxn_string[-len(old_comp):]:
                rxn.id = rxn.id.replace("_" + old_comp, "_" + new_comp)
                if change_name and '_' + old_comp in rxn.name:
                    rxn.name = rxn.name.replace("_" + old_comp, "_" + new_comp)

        if inplace:
            return
        else:
            return model

    def rename_compartment(self, model, rename):
        old_compartments = model.compartments
        for comp in rename.keys():
            assert comp in old_compartments

        for met in model.metabolites:
            if met.compartment in rename.keys():
                met.compartment = rename[met.compartment]
        model.repair()

        for old_comp, new_comp in rename.items():
            self.rename_comp_in_met_id(model, old_comp, new_comp, inplace=True, change_name=False,
                                       remove_comp_from_name=True)
            self.rename_comp_in_rxn_id(model, old_comp, new_comp, inplace=True, change_name=False)

        model.repair()
        return model

    def add_exchange_reactions_to_compartment(self, model, comp, inplace=True):
        if not inplace:
            model = model.copy()

        for met in model.metabolites.query(lambda x: x.compartment == comp):
            self.force_add_exchange(model, met)

        model.repair()

        if inplace:
            return
        else:
            return model

    def add_exchange_reactions_to_metabolites(self, model, mets, lb=0., inplace=True):
        if not inplace:
            model = model.copy()

        for met in mets:
            model.add_boundary(met, type="exchange", lb=lb)

        model.repair()

        if inplace:
            return
        else:
            return model

    def force_add_exchange(self,
                           model,
                           metabolite,
                           reaction_id=None,
                           lb=None,
                           ub=None,
                           sbo_term=None,
                           ):

        sbo_terms = cobra.medium.sbo_terms
        ub = 1000. if ub is None else ub
        lb = -1000. if lb is None else lb
        types = {
            "exchange": ("EX", lb, ub, sbo_terms["exchange"]),
            "demand": ("DM", 0, ub, sbo_terms["demand"]),
            "sink": ("SK", lb, ub, sbo_terms["sink"]),
        }
        rxn_type = "exchange"
        if rxn_type in types:
            prefix, lb, ub, default_term = types[rxn_type]
            if reaction_id is None:
                reaction_id = "{}_{}".format(prefix, metabolite.id)
            if sbo_term is None:
                sbo_term = default_term
        if reaction_id is None:
            raise ValueError(
                "Custom types of boundary reactions require a custom "
                "identifier. Please set the `reaction_id`."
            )
        if reaction_id in model.reactions:
            raise ValueError(
                "Boundary reaction '{}' already exists.".format(reaction_id)
            )
        name = "{} {}".format(metabolite.name, rxn_type)
        rxn = cobra.Reaction(id=reaction_id, name=name, lower_bound=lb, upper_bound=ub)
        rxn.add_metabolites({metabolite: -1})
        if sbo_term:
            rxn.annotation["sbo"] = sbo_term
        model.add_reactions([rxn])
        return rxn

    def find_metabolites_without_exchange_rxn(self, model, exchg_comp=""):
        if len(exchg_comp) == 0:
            exchg_comp = cobra.medium.find_external_compartment(model)
        comp_mets = model.metabolites.query(lambda x: x.compartment == exchg_comp)
        exchg_mets = [list(rxn.metabolites)[0] for rxn in model.exchanges]
        mets_without_exchg = list((set(comp_mets) - set(exchg_mets)) - set([self.biomass_met]))
        return mets_without_exchg

    def convert_exchange_to_transport_reaction(self, model, old_comp, inplace=True):
        if not inplace:
            model = model.copy()

        for rxn in model.exchanges:
            rxn_string = rxn.id
            if old_comp in rxn_string and old_comp == rxn_string[-len(old_comp):]:
                # Replace the exchange reaction term with TP_
                if "EX_" in rxn.id:
                    rxn.id = rxn.id.replace("EX_", "TP_")
                # Remove the SBO term for exchange reaction
                if "sbo" in rxn.annotation and "SBO:0000627" in rxn.annotation["sbo"]:
                    rxn.annotation.pop("sbo")
                out_met_rxn = rxn.copy()
                new_met_stoich = {}
                for met, stoich in out_met_rxn.metabolites.items():
                    new_met_stoich[model.metabolites.get_by_id(self._exchange_met_name_conversion[met.id])] = -stoich
                rxn.add_metabolites(new_met_stoich)
                if "Exchange" in rxn.name:
                    rxn.name = rxn.name.replace("Exchange", "Transport")

        model.repair()

        if inplace:
            return
        else:
            return model

    def add_boundary_metabolites_to_exchange_compartment(self, model, new_comp="exchg", old_comp="", inplace=True):
        if not inplace:
            model = model.copy()

        if not old_comp:
            old_comp = cobra.medium.find_external_compartment(model)

        for met in model.metabolites:
            if met == self.biomass_met:
                continue
            met_string = met.id
            if old_comp in met_string and old_comp == met_string[-len(old_comp):]:
                new_met = model.metabolites.get_by_id(met_string).copy()
                if self._name_via_annotation is not None and self._name_via_annotation in new_met.annotation.keys():
                    annotation_id = new_met.annotation[self._name_via_annotation]
                    if isinstance(annotation_id, list) and len(annotation_id) == 1:
                        annotation_id = annotation_id[0]
                    elif not isinstance(annotation_id, str):
                        raise AssertionError(
                            f"Annotation for merging contains multiple IDs! Cannot merge {annotation_id}")
                    new_met.id = make_string_sbml_id_compatible(annotation_id + "_" + new_comp,
                                                                remove_ascii_escapes=True)
                else:
                    new_met.id = new_met.id.replace("_" + old_comp, "_" + new_comp)
                self._exchange_met_name_conversion[met_string] = new_met.id
                if met.name[-(len(old_comp)):] == old_comp:
                    new_met.name = met.name[:-(len(old_comp))] + new_comp
                new_met.compartment = new_comp
                model.add_metabolites([new_met])

        model.repair()

        if inplace:
            return
        else:
            return model

    def ensure_compartment_suffix(self, model, inplace=True):
        if not inplace:
            model = model.copy()

        for met in model.metabolites:
            comp = met.compartment
            if comp not in met.id or comp != met.id[-len(comp):]:
                # Suffix with compartment name
                met.id += f"_{comp}"

        for rxn in model.reactions:
            met_comps = [met.compartment for met in rxn.metabolites.keys()]
            if len(set(met_comps)) == 1:  # Only 1 compartment
                comp = met_comps[0]
                if comp not in rxn.id or comp != rxn.id[-len(comp):]:
                    # Suffix with compartment name
                    rxn.id += f"_{comp}"

        return model

    def add_exchange_compartment(self, model, exchg_comp_name="exchg", add_missing_transports=False, inplace=True):
        if not inplace:
            model = model.copy()
        old_exc_comp = cobra.medium.find_external_compartment(model)
        # Add metabolites
        self.add_boundary_metabolites_to_exchange_compartment(model, new_comp=exchg_comp_name, inplace=True)
        # Add transport reactions
        if add_missing_transports:
            mets_without_exchg = self.find_metabolites_without_exchange_rxn(model)
            self.add_exchange_reactions_to_metabolites(model, mets_without_exchg, inplace=True)
        self.convert_exchange_to_transport_reaction(model, old_exc_comp, inplace=True)
        # Add exchange reactions
        self.add_exchange_reactions_to_compartment(model, exchg_comp_name, inplace=True)

        return model

    def prefix_metabolite_names(self, model, prefix, exclude_compartment="", inplace=True):
        if not inplace:
            model = model.copy()  # Don't want to change the original

        for metabolite in model.metabolites:
            if not metabolite.compartment == exclude_compartment:
                metabolite.id = f"{prefix}{metabolite.id}"

        model.repair()

        return model

    def prefix_reaction_names(self, model, prefix, exclude_compartment="", inplace=True):
        if not inplace:
            model = model.copy()  # Don't want to change the original

        for reaction in model.reactions:
            if len(exclude_compartment) > 0 and reaction.id[-len(exclude_compartment):] != exclude_compartment:
                reaction.id = f"{prefix}{reaction.id}"

        model.repair()

        return model

    def remove_biomass_exchange_rxn(self, model, remove_all_consuming_rxns=True):
        for reaction in self.biomass_met.reactions:
            if reaction in model.exchanges:
                reaction.remove_from_model()
            elif remove_all_consuming_rxns and self.biomass_met in reaction.reactants:
                reaction.remove_from_model(remove_orphans=True)
        return

    def prepare_for_merging(self):
        self.prepared_model = self.model.copy()
        self.biomass_met = get_model_biomass_compound(self.prepared_model, generate_if_none=True)
        self.remove_biomass_exchange_rxn(self.prepared_model)

        # Remove ascii escape characters from sbml ids, as they are not compatible
        make_model_ids_sbml_conform(self.prepared_model)

        # Check that compartment names are in metabolite and reaction ids
        self.ensure_compartment_suffix(self.prepared_model)

        rename = {}
        for comp in self.prepared_model.compartments:
            rename[comp] = self.name + "_" + comp
        self.rename_compartment(self.prepared_model, rename)
        self.add_exchange_compartment(self.prepared_model, add_missing_transports=True)
        self.prefix_metabolite_names(self.prepared_model, self.name + "_", exclude_compartment="exchg")
        self.prefix_reaction_names(self.prepared_model, self.name + "_", exclude_compartment="exchg")
        return self.prepared_model


class CommunityModel:
    """
    This class contains a single organism metabolic model and its meta information needed for community model generation
    """
    models: List[SingleOrganismModel]
    name: str
    medium_flag: bool = False
    abundance_flag: bool = False
    _unconstrained_model: cobra.Model = None
    _community_model: cobra.Model = None
    _medium: dict = None
    _merge_via_annotation: str = None
    _abundance_dict: dict = None
    _member_names = None

    def __init__(self, models=None, name="", merge_via_annotation=None, **kwargs):
        self.models = models

        if models is not None:
            model_names = [model.name for model in self.models]
        elif "member_names" in kwargs.keys():  # Construction from saved file
            model_names = kwargs["member_names"]
        else:
            raise AssertionError("No models provided to CommunityModel object!")

        if not list_contains_unique_strings(model_names):
            raise AssertionError(f"Model names contain duplicates!")
        if list_of_strings_is_self_contained(model_names):
            raise AssertionError(f"Some model names are contained in others!")

        self._member_names = model_names

        self.name = make_string_sbml_id_compatible(name)
        if name != self.name:
            print(f"Warning: model name {name} is not compliant with sbml id standards and was changed to {self.name}")
        if merge_via_annotation is not None:
            self._merge_via_annotation = merge_via_annotation
            for model in self.models:
                model.set_name_via_annotation(merge_via_annotation)

        if "unconstrained_model" in kwargs.keys():
            self._unconstrained_model = kwargs["unconstrained_model"]

        if "abundance_profile" in kwargs.keys():
            self.apply_abundance(kwargs["abundance_profile"])

    @property
    def unconstrained_model(self):
        return self._unconstrained_model

    @unconstrained_model.getter
    def unconstrained_model(self):
        if self._unconstrained_model is None:
            print(f"No unconstrained community model generated yet. Generating now:")
            self.generate_community_model()
            print(f"Generated unconstrained community model.")
        return self._unconstrained_model

    @property
    def community_model(self):
        return self._community_model

    @community_model.getter
    def community_model(self):
        if self._community_model is None:
            print(f"No constrained community model set yet. Using the unconstrained model instead.")
            self._community_model = self.unconstrained_model
            self.abundance_flag = False
            self.medium_flag = False
        return self._community_model

    @property
    def medium(self):
        return self._medium

    @medium.getter
    def medium(self):
        if self._medium is None:
            raise AssertionError("Error: No medium set for this community model.\nPlease set the medium with "
                                 ".load_medium_from_file('/path/to/medium_file.csv')")
        return self._medium

    @medium.setter
    def medium(self, medium_dict):
        # Check that dataframe has the correct format
        try:
            assert all([isinstance(key, str) for key in medium_dict.keys()])
            assert all([isinstance(value, float) for value in medium_dict.values()])
        except AssertionError:
            raise AssertionError
        self._medium = medium_dict

    def generate_member_name_conversion_dict(self):
        conversion_dict = {}
        for member in self.models:
            old_name, new_name = member.get_name_conversion()
            conversion_dict[old_name] = new_name
        return conversion_dict

    def get_member_names(self):
        if self.models is not None:
            member_names = [member.name for member in self.models]
        else:
            member_names = self._member_names
        return member_names

    def get_unbalanced_reactions(self):
        return cobra.manipulation.validate.check_mass_balance(self.community_model)

    def is_mass_balanced(self):
        return not bool(self.get_unbalanced_reactions())

    def generate_community_model(self):
        merged_model = None
        biomass_mets = []
        idx = 0
        for model in self.models:
            idx += 1
            if idx == 1:
                merged_model = model.prepare_for_merging()
                biomass_met_id = model.biomass_met.id
                biomass_mets.append(merged_model.metabolites.get_by_id(biomass_met_id))
            else:
                extended_model = model.prepare_for_merging()
                unbalanced_metabolites = check_mass_balance_of_metabolites_with_identical_id(extended_model, merged_model)
                for met_id in unbalanced_metabolites:
                    met_base_name = get_metabolite_id_without_compartment(extended_model.metabolites.get_by_id(met_id))
                    print(f"WARNING: matching of the metabolite {met_base_name} is unbalanced (mass and/or charge). "
                          f"Please manually curate this metabolite for a mass and charge balanced model!")
                no_annotation_overlap = check_annotation_overlap_of_metabolites_with_identical_id(extended_model,
                                                                                             merged_model)
                for met_id in no_annotation_overlap:
                    met_base_name = get_metabolite_id_without_compartment(extended_model.metabolites.get_by_id(met_id))
                    print(f"WARNING: no annotation overlap found for matching metabolite {met_base_name}. "
                          f"Please make sure that the metabolite with this ID is indeed representing the same substance"
                          f" in all models!")
                merged_model.merge(extended_model)
                biomass_met_id = model.biomass_met.id
                biomass_mets.append(merged_model.metabolites.get_by_id(biomass_met_id))

        biomass_met = cobra.Metabolite("cpd11416_exchg", name='Community Biomass', compartment='exchg')
        merged_model.add_metabolites([biomass_met])
        biomass_rxn = cobra.Reaction("community_biomass")
        biomass_rxn.name = "Community Biomass Reaction"
        biomass_rxn.lower_bound = 0.
        biomass_rxn.upper_bound = 1000.
        rxn_mets = {}
        for met in biomass_mets:
            rxn_mets[met] = -1.
        rxn_mets[biomass_met] = 1.
        biomass_rxn.add_metabolites(rxn_mets)
        merged_model.add_reactions([biomass_rxn])
        merged_model.add_boundary(merged_model.metabolites.get_by_id("cpd11416_exchg"), type="exchange", lb=0.)
        merged_model.objective = "community_biomass"

        # Remove old biomass reactions
        old_biomass_exchange_rxns = []
        for met in biomass_mets:
            for rxn in met.reactions:
                if rxn in merged_model.exchanges:
                    old_biomass_exchange_rxns.append(rxn)
        merged_model.remove_reactions(old_biomass_exchange_rxns, remove_orphans=True)
        make_model_ids_sbml_conform(merged_model)
        merged_model.id = self.name

        self._unconstrained_model = merged_model

        if not self.is_mass_balanced():
            print("WARNING: Not all reactions in the model are mass and charge balanced. To check which reactions are imbalanced, please run the get_unbalanced_reactions method of this CommunityModel object")

        return merged_model

    def apply_abundance(self, abd_dict):
        # Check if organism names are in the model
        try:
            assert all([name in self.get_member_names() for name in abd_dict.keys()])
        except AssertionError:
            print(f"Error: Some names in the abundances are not part of the model.")
            print(f"\tAbundances: {abd_dict.keys()}")
            print(f"\tOrganisms in model: {self.get_member_names()}")
            raise AssertionError

        # Check that abundances sum to 1
        try:
            assert np.isclose([sum(abd_dict.values())], [1.])
        except AssertionError:
            print(f"Warning: Abundances do not sum up to 1. Correction will be applied")
            if sum(abd_dict.values()) == 0.:
                print(f"Error: The sum of abundances is 0")
                raise ValueError
            correction_factor = 1 / sum(abd_dict.values())
            for name, abundance in abd_dict.items():
                abd_dict[name] = abundance * correction_factor
            print(f"Correction applied. New abundances are:\n{abd_dict}")
            assert np.isclose([sum(abd_dict.values())], [1.])

        # Extend abundances to include all organisms of model
        for name in self.get_member_names():
            if name not in abd_dict.keys():
                abd_dict[name] = 0.

        # Apply abundances to model
        abd_model = self.unconstrained_model.copy()
        for rxn in abd_model.reactions:
            for name in self.get_member_names():
                if rxn.id.find(name) == 0:
                    rxn.lower_bound *= float(abd_dict[name])
                    rxn.upper_bound *= float(abd_dict[name])

        # Change biomass reaction
        biomass_rxn = abd_model.reactions.get_by_id("community_biomass")
        stoichiometry = biomass_rxn.metabolites
        for met, value in stoichiometry.items():
            if value > 0:
                continue
            else:
                for name in self.get_member_names():
                    if met.id.find(name) == 0:
                        stoichiometry[met] = -float(abd_dict[name])
        biomass_rxn.add_metabolites(stoichiometry, combine=False)

        self._community_model = abd_model
        self.abundance_flag = True
        self._abundance_dict = abd_dict
        if self.medium_flag:
            self.apply_medium()
        return abd_model

    def equal_abundance(self):
        abundances = {}
        names = self.get_member_names()
        for name in names:
            abundances[name] = 1. / len(names)
        return self.apply_abundance(abundances)

    def load_medium_from_file(self, file_path):
        # load the medium dictionary
        medium_dict = read_medium_from_file(file_path, comp="_exchg")
        self.medium = medium_dict

    def apply_medium(self):
        test_if_medium_exists = self.medium
        medium_model = self.community_model.copy()
        # Exclude metabolites from the medium that are not part of the model
        medium_refined = {}
        for rxn in self.medium.keys():
            try:
                medium_model.reactions.get_by_id(rxn)
                medium_refined[rxn] = self.medium[rxn]
            except KeyError:
                # Exclude reactions that are not part of the model
                pass

        # Set the medium
        medium_model.medium = medium_refined

        self._community_model = medium_model
        self.medium_flag = True
        return medium_model

    def run_fba(self, unconstrained=False):
        if unconstrained:
            solution = self.unconstrained_model.optimize()
        else:
            solution = self.community_model.optimize()
        solution_df = solution.fluxes.to_frame()
        solution_df.insert(loc=0, column='reaction', value=list(solution.fluxes.index))
        solution_df.columns = ["reaction_id", "flux"]

        return solution_df

    def fba_solution_flux_vector(self, file_path="", unconstrained=False):
        solution_df = self.run_fba(unconstrained=unconstrained)

        if len(file_path) > 0:
            print(f"Saving flux vector to {file_path}")
            solution_df.to_csv(file_path, sep="\t", header=True, index=False)
        return solution_df

    def run_fva(self, unconstrained=False, fraction_of_optimum=0.9, composition_agnostic=False):
        if unconstrained:
            model = self.unconstrained_model
        else:
            model = self.community_model

        reactions = model.reactions.query(lambda x: any([met.compartment == "exchg" for met in x.metabolites.keys()]))

        if composition_agnostic:
            with model as composition_agnostic_model:
                for reaction in composition_agnostic_model.reactions:
                    if reaction.lower_bound > 0.:
                        reaction.lower_bound = 0.
                    if reaction.upper_bound < 0.:
                        reaction.upper_bound = 0.
                solution_df = cobra.flux_analysis.flux_variability_analysis(composition_agnostic_model, reactions,
                                                                            fraction_of_optimum=fraction_of_optimum)
        else:
            solution_df = cobra.flux_analysis.flux_variability_analysis(model, reactions,
                                                                        fraction_of_optimum=fraction_of_optimum)

        solution_df.insert(loc=0, column='reaction', value=list(solution_df.index))
        solution_df.columns = ["reaction_id", "min_flux", "max_flux"]

        return solution_df

    def fva_solution_flux_vector(self, file_path="", unconstrained=False, fraction_of_optimum=0.9):
        solution_df = self.run_fva(unconstrained=unconstrained, fraction_of_optimum=fraction_of_optimum)

        if len(file_path) > 0:
            print(f"Saving flux vector to {file_path}")
            solution_df.to_csv(file_path, sep="\t", header=True, index=False)
        return solution_df

    def cross_feeding_metabolites_from_fba(self, unconstrained=False):
        if unconstrained:
            model = self.unconstrained_model
        else:
            model = self.community_model

        solution_df = self.run_fba(unconstrained=unconstrained)
        rows = []

        exchg_metabolites = model.metabolites.query(lambda x: x.compartment == "exchg")
        member_names = self.get_member_names()

        for exchg_met in exchg_metabolites:
            # Check flux of transport ("_TP_") reactions to organism
            row_dict = {"metabolite_id": exchg_met.id, "metabolite_name": exchg_met.name}
            for name in member_names:
                row_dict[name] = 0.
            for rxn in exchg_met.reactions:
                if "_TP_" not in rxn.id:
                    continue
                rxn_member = rxn.id.split("_TP_")[0]
                assert rxn_member in member_names
                assert rxn.id in set(solution_df["reaction_id"])
                flux = float(solution_df.loc[rxn.id, "flux"])
                row_dict[rxn_member] = 0. if close_to_zero(flux) else flux
            rows.append(row_dict)

        exchg_metabolite_df = pd.DataFrame(rows)
        cross_feeding_metabolites = exchg_metabolite_df.copy()
        cross_feeding_metabolites.drop(columns=["metabolite_id", "metabolite_name"], inplace=True)
        cross_feeding_mask = cross_feeding_metabolites.apply(lambda x: any(x > 0.) and any(x < 0.), axis=1)
        exchg_metabolite_df["cross_feeding"] = cross_feeding_mask
        return exchg_metabolite_df

    def cross_feeding_metabolites_from_fva(self, unconstrained=False, fraction_of_optimum=0., composition_agnostic=False):
        if unconstrained:
            model = self.unconstrained_model
        else:
            model = self.community_model

        solution_df = self.run_fva(unconstrained=unconstrained, fraction_of_optimum=fraction_of_optimum, composition_agnostic=composition_agnostic)
        rows = []
        exchg_metabolites = model.metabolites.query(lambda x: x.compartment == "exchg")
        member_names = self.get_member_names()

        for exchg_met in exchg_metabolites:
            # Check flux of transport ("_TP_") reactions to organism
            row_dict = {"metabolite_id": exchg_met.id, "metabolite_name": exchg_met.name, "cross_feeding": False}
            for name in member_names:
                row_dict[name + "_min_flux"] = 0.
                row_dict[name + "_max_flux"] = 0.
            for rxn in exchg_met.reactions:
                if "_TP_" not in rxn.id:
                    continue
                rxn_member = rxn.id.split("_TP_")[0]
                assert rxn_member in member_names
                assert rxn.id in set(solution_df["reaction_id"])
                min_flux = float(solution_df.loc[rxn.id, "min_flux"])
                max_flux = float(solution_df.loc[rxn.id, "max_flux"])
                row_dict[rxn_member + "_min_flux"] = 0. if close_to_zero(min_flux) else min_flux
                row_dict[rxn_member + "_max_flux"] = 0. if close_to_zero(max_flux) else max_flux
            interaction = False
            for name in member_names:
                if row_dict[name + "_min_flux"] < 0. and any(
                        [row_dict[other + "_max_flux"] > 0. for other in list_without_element(member_names, name)]):
                    interaction = True
                    break
            row_dict["cross_feeding"] = interaction
            rows.append(row_dict)

        exchg_metabolite_df = pd.DataFrame(rows)

        return exchg_metabolite_df

    def format_exchg_rxns(self, exchg_metabolite_df):
        rows = []
        member_names = self.get_member_names()

        for idx, row in exchg_metabolite_df.iterrows():
            row_dict = {"metabolite_id": row["metabolite_id"], "metabolite_name": row["metabolite_name"], "cross_feeding": row["cross_feeding"], "produced_by": [], "consumed_by": []}
            for member in member_names:
                if row[member + "_min_flux"] < 0.:
                    row_dict["consumed_by"].append(member)
                if row[member + "_max_flux"] > 0.:
                    row_dict["produced_by"].append(member)
            rows.append(row_dict)
        exchg_metabolite_df = pd.DataFrame(rows)

        return exchg_metabolite_df

    def potential_metabolite_exchanges(self):
        exchange_fva_df = self.cross_feeding_metabolites_from_fva(unconstrained=False, fraction_of_optimum=0.,
                                                                  composition_agnostic=True)
        return self.format_exchg_rxns(exchange_fva_df)

    def save(self, file_path):
        """
        Save the community model object as a SBML file. This also includes the names of the community members and their abundance (if set).
        """
        # Generate a libsbml.model object
        cobra.io.write_sbml_model(self._unconstrained_model, filename=file_path)
        sbml_doc = cobra.io.sbml._get_doc_from_filename(file_path)
        sbml_model = sbml_doc.getModel()

        # Add parameters for the community members and their abundance
        abundances = {}
        if self._abundance_dict is None:
            for organism in self.get_member_names():
                abundances[organism] = None
        else:
            abundances = self._abundance_dict

        for member, fraction in abundances.items():
            create_abundance_parameter(sbml_model=sbml_model, member_id=member, abundance=fraction)

        libsbml.writeSBMLToFile(sbml_doc, file_path)

        return

    @classmethod
    def load(cls, file_path):

        abundance_parameters = get_abundance_parameters_from_sbml_file(file_path)
        assert len(abundance_parameters) > 0

        constructor_args = {}
        constructor_args["member_names"] = list(abundance_parameters.keys())
        if any([val is not None for val in abundance_parameters.values()]):
            constructor_args["abundance_profile"] = abundance_parameters
        constructor_args["unconstrained_model"] = cobra.io.read_sbml_model(file_path)
        name = constructor_args["unconstrained_model"].id
        return cls(name=name, **constructor_args)



def doall(model_folder="", models=[], community_name="community_model", abundance="equal", medium=None,
          fba_solution_path=None, fva_solution_path=None, fva_solution_threshold=0.9, fba_interaction_path=None,
          fva_interaction_path=None, fva_interaction_threshold=0.9, sbml_output_path=None, return_as_cobra_model=False,
          merge_via_annotation=None):
    # Load singel organism models
    # Either from folder or as list of file names or list of cobra models
    if model_folder != "":
        named_models = load_named_models_from_dir(model_folder)
    elif not isinstance(models, list) or len(models) == 0:
        raise ValueError(f"No models supplied to the doall function. Please supply either a path to the folder "
                         f"containing the models to the model_folder variable or a list of filepaths or cobra models "
                         f"to the models variable.")
    elif all(list(map(lambda x: isinstance(x, cobra.Model), models))):
        # Extract names and store in named models
        named_models = {model.name: model for model in models}
    elif all(list(map(lambda x: isinstance(x, cobra.Model), models))):
        named_models = {}
        for model_path in models:
            model, name = load_named_model(model_path)
            named_models[name] = model
    else:
        raise TypeError(f"Input models are either of mixed type or neither filepath, nor cobra model.")

    # Create single organism models
    single_org_models = [SingleOrganismModel(model, name) for name, model in named_models.items()]

    # Create a community model
    com_model_obj = CommunityModel(single_org_models, community_name, merge_via_annotation=merge_via_annotation)
    com_model_obj.generate_community_model()

    # Apply abundance (either None, "equal", or an abundance dict
    if abundance == "equal":
        com_model_obj.equal_abundance()
    elif isinstance(abundance, dict):
        name_conversion = com_model_obj.generate_member_name_conversion_dict()
        tmp_abundance = {}
        for name, fraction in abundance:
            tmp_abundance[name_conversion[name]] = fraction
        com_model_obj.apply_abundance(tmp_abundance)
    else:
        pass  # No abundance to apply

    # Apply medium
    if medium is not None:
        com_model_obj.load_medium_from_file(medium)
        com_model_obj.apply_medium()

    if sbml_output_path is not None:
        cobra.io.write_sbml_model(com_model_obj.community_model, filename=sbml_output_path)

    if fba_solution_path is not None:
        try:
            com_model_obj.fba_solution_flux_vector(file_path=fba_solution_path)
        except cobra.exceptions.Infeasible:
            print(f"WARNING: FBA of community is infeasible. No FBA flux vector file was generated.")

    if fva_solution_path is not None:
        try:
            com_model_obj.fva_solution_flux_vector(file_path=fva_solution_path, fraction_of_optimum=0.0)
        except cobra.exceptions.Infeasible:
            print(f"WARNING: FVA of community is infeasible. No FVA flux vector file was generated.")

    if return_as_cobra_model:
        # Retrieve community model
        return com_model_obj.community_model
    else:
        return com_model_obj  # Return the community model object


if __name__ == "__main__":
    # get the path of this script and add it to the "pythonpath"
    SCRIPT_PATH = os.path.split(os.path.realpath(os.path.abspath(__file__)))[0]
    sys.path.insert(0, SCRIPT_PATH)

    parser = create_arg_parser()
    args = parser.parse_args()
    args = check_args(args)
    print(args)
    if args.abundance is not None and args.abundance != "equal":
        # Retrieve the abundance from file
        pass
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        doall(model_folder=args.input[0], community_name=args.name, abundance=args.abundance, medium=args.medium,
              fba_solution_path=args.fba_solution_path, fva_solution_path=args.fva_solution_path,
              fva_solution_threshold=args.fva_flux, fba_interaction_path=args.fba_interaction_path,
              fva_interaction_path=args.fva_interaction_path, fva_interaction_threshold=args.fva_interaction,
              sbml_output_path=args.sbml_output_path, return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation)
    else:
        doall(models=args.input, community_name=args.name, abundance=args.abundance, medium=args.medium,
              fba_solution_path=args.fba_solution_path, fva_solution_path=args.fva_solution_path,
              fva_solution_threshold=args.fva_flux, fba_interaction_path=args.fba_interaction_path,
              fva_interaction_path=args.fva_interaction_path, fva_interaction_threshold=args.fva_interaction,
              sbml_output_path=args.sbml_output_path, return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation)

    print("All done!")
    sys.exit(0)
