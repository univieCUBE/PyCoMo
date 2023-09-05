#!/usr/bin/env python3

"""
Authors: Michael Predl, Marianne Mießkes
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
from math import inf
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
    shared_compartment_name = None
    _name_via_annotation: str = None
    _exchange_met_name_conversion: dict = {}

    def __init__(self, model, name, biomass_met_id="", name_via_annotation=None, shared_compartment_name="medium"):
        self.model = model.copy()
        self._original_name = name
        self.name = make_string_sbml_id_compatible(name, remove_ascii_escapes=True)
        if name != self.name:
            print(f"Warning: model name {name} is not compliant with sbml id standards and was changed to {self.name}")
        self.biomass_met_id = biomass_met_id
        self._name_via_annotation = name_via_annotation
        self.shared_compartment_name = shared_compartment_name

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
        mets_without_exchg = list((set(comp_mets) - set(exchg_mets)) - {self.biomass_met})
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

    def add_boundary_metabolites_to_exchange_compartment(self, model, new_comp=None, old_comp="", inplace=True):
        if new_comp is None:
            new_comp = self.shared_compartment_name

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

    def add_exchange_compartment(self, model, exchg_comp_name=None, add_missing_transports=False, inplace=True):
        if exchg_comp_name is None:
            exchg_comp_name = self.shared_compartment_name

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
                if prefix != metabolite.id[:len(prefix)]:
                    metabolite.id = f"{prefix}{metabolite.id}"

        model.repair()

        return model

    def prefix_reaction_names(self, model, prefix, exclude_compartment="", inplace=True):
        if not inplace:
            model = model.copy()  # Don't want to change the original

        for reaction in model.reactions:
            if len(exclude_compartment) > 0 and reaction.id[-len(exclude_compartment):] != exclude_compartment:
                if prefix != reaction.id[:len(prefix)]:
                    reaction.id = f"{prefix}{reaction.id}"

        model.repair()

        return model

    def prefix_gene_names(self, model, prefix, inplace=True):
        if not inplace:
            model = model.copy()  # Don't want to change the original

        rename_dict = {}
        for gene in model.genes:
            if not gene.name:
                gene.name = gene.id
            if prefix != gene.id[:len(prefix)]:
                rename_dict[gene.id] = f"{prefix}{gene.id}"

        if rename_dict:
            cobra.manipulation.modify.rename_genes(model, rename_dict)

        model.repair()

        return model

    def remove_biomass_exchange_rxn(self, model, remove_all_consuming_rxns=True):
        for reaction in self.biomass_met.reactions:
            if reaction in model.exchanges:
                reaction.remove_from_model()
            elif remove_all_consuming_rxns and self.biomass_met in reaction.reactants:
                reaction.remove_from_model(remove_orphans=True)
        return

    def prepare_for_merging(self, shared_compartment_name=None):
        if shared_compartment_name is not None:
            self.shared_compartment_name = shared_compartment_name
        self.prepared_model = self.model.copy()
        self.biomass_met = get_model_biomass_compound(self.prepared_model, self.shared_compartment_name,
                                                      generate_if_none=True)
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
        self.prefix_metabolite_names(self.prepared_model, self.name + "_",
                                     exclude_compartment=self.shared_compartment_name)
        self.prefix_reaction_names(self.prepared_model, self.name + "_",
                                   exclude_compartment=self.shared_compartment_name)
        self.prefix_gene_names(self.prepared_model, self.name + "_")
        return self.prepared_model


class CommunityModel:
    """
    This class contains a single organism metabolic model and its meta information needed for community model generation
    """
    models: List[SingleOrganismModel]
    name: str
    medium_flag: bool = False
    abundance_flag: bool = False
    fraction_reaction_flag: bool = False
    mu_c: float = 1.
    fixed_abundance_flag: bool = False
    fixed_growth_rate_flag: bool = False
    max_flux: float = 1000.
    shared_compartment_name: str = None
    _f_metabolites: list = None
    _f_reactions: list = None
    _unconstrained_model: cobra.Model = None
    _community_model: cobra.Model = None
    _medium: dict = None
    _merge_via_annotation: str = None
    _abundance_dict: dict = None
    _member_names: list = None
    _backup_metabolites: dict = {}

    def __init__(self, models=None, name="", merge_via_annotation=None, mu_c=1., fraction_flag=True, max_flux=1000.,
                 shared_compartment_name="medium", **kwargs):
        self.models = models
        self.fraction_reaction_flag = fraction_flag
        self.mu_c = mu_c
        self.shared_compartment_name = shared_compartment_name

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

        if max_flux > 0.:
            self.max_flux = max_flux
        else:
            print(f"Warning: maximum flux value is not greater than 0 ({max_flux}). Using default value of 1000.0 "
                  f"instead.")
            self.max_flux = 1000.

        self.name = make_string_sbml_id_compatible(name)
        if name != self.name:
            print(f"Warning: model name {name} is not compliant with sbml id standards and was changed to {self.name}")
        if merge_via_annotation is not None:
            self._merge_via_annotation = merge_via_annotation
            for model in self.models:
                model.set_name_via_annotation(merge_via_annotation)

        if "fixed_abundance" in kwargs.keys():
            self.fixed_abundance_flag = kwargs["fixed_abundance"]

        if "fixed_growth_rate" in kwargs.keys():
            self.fixed_growth_rate_flag = kwargs["fixed_growth_rate"]

        if "unconstrained_model" in kwargs.keys():
            self._unconstrained_model = kwargs["unconstrained_model"]
            for member in self._member_names:
                try:
                    met = self._unconstrained_model.metabolites.get_by_id(f"{member}_f_biomass_met")
                except KeyError:
                    met = cobra.Metabolite(f'{member}_f_biomass_met', name=f'Fraction Biomass Metabolite of {member}',
                                           compartment='fraction_reaction')
                self._backup_metabolites[f"{member}_f_biomass_met"] = met

        if "abundance_profile" in kwargs.keys():
            self._abundance_dict = kwargs["abundance_profile"]

        if "shared_compartment_name" in kwargs.keys():
            self.shared_compartment_name = kwargs["shared_compartment_name"]

    @property
    def unconstrained_model(self):
        # TODO: remove the unconstrained model, as it is not needed anymore.
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
    def f_metabolites(self):
        return self._f_metabolites

    @f_metabolites.getter
    def f_metabolites(self):
        self._f_metabolites = self.community_model.metabolites.query(lambda x: x.compartment == "fraction_reaction")
        if self._f_metabolites is None:
            self._f_metabolites = []
        return self._f_metabolites

    @property
    def f_reactions(self):
        return self._f_reactions

    @f_reactions.getter
    def f_reactions(self):
        self._f_reactions = self.community_model.reactions.query(
            lambda x: (x.id[:3] == "SK_" and x.id[-3:] in {"_lb", "_ub"}) or "_fraction_reaction" in x.id)
        if self._f_reactions is None:
            self._f_reactions = []
        return self._f_reactions

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

    def summary(self, suppress_f_metabolites=True):
        summary = self.community_model.summary()

        if suppress_f_metabolites:
            model = self.community_model
            new_secretion_flux_rows = []
            old_secretion_flux = summary.secretion_flux
            for idx, row in old_secretion_flux.iterrows():
                if model.metabolites.get_by_id(row["metabolite"]).compartment != "fraction_reaction":
                    new_secretion_flux_rows.append(row)

            new_secretion_flux = pd.DataFrame(new_secretion_flux_rows)
            summary.secretion_flux = new_secretion_flux

        return summary

    def generate_member_name_conversion_dict(self):
        conversion_dict = {}
        if self.models is not None:
            for member in self.models:
                old_name, new_name = member.get_name_conversion()
                conversion_dict[old_name] = new_name
        else:
            print("Warning: There are no member models in the community model object.")
            for member_name in self.get_member_names():
                conversion_dict[member_name] = member_name
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

    def get_loops(self):
        """This is a function to find closed loops that can sustain flux without any input or output. Such loops are
        thermodynamically infeasible and biologically nonsensical. Users should be aware of their presence and
        either remove them or check any model solutions for the presence of these cycles."""
        try:
            original_medium = self.medium
        except AssertionError:
            original_medium = self.community_model.medium
        no_medium = {}
        self.community_model.medium = no_medium

        solution_df = find_loops_in_model(self.convert_to_model_without_fraction_metabolites())

        self.community_model.medium = original_medium
        return solution_df[
            (~ solution_df["min_flux"].apply(close_to_zero)) | (~ solution_df["max_flux"].apply(close_to_zero))]

    def get_member_name_of_reaction(self, reaction):
        """
        This function will return the name of the member the reaction belongs to by extracting this information from its
         ID.
        """
        if isinstance(reaction, str):
            metabolite = self.community_model.reactions.get_by_id(reaction)

        member_name = None

        for name in self.get_member_names():
            if name == reaction.id[:len(name)]:
                member_name = name
                break

        return member_name

    def get_member_name_of_metabolite(self, metabolite):
        """
        This function will return the name of the member the metabolite belongs to by extracting this information from
        its ID.
        """
        if isinstance(metabolite, str):
            metabolite = self.community_model.metabolites.get_by_id(metabolite)

        member_name = None

        for name in self.get_member_names():
            if name == metabolite.id[:len(name)]:
                member_name = name
                break

        return member_name

    def get_member_name_of_compartment(self, compartment):
        """
        This function will return the name of the member the compartment belongs to by extracting this information from
        its ID.
        """
        member_name = None

        for name in self.get_member_names():
            if name == compartment[:len(name)]:
                member_name = name
                break

        return member_name

    def generate_community_model(self):
        merged_model = None
        biomass_mets = {}
        idx = 0
        for model in self.models:
            idx += 1
            if idx == 1:
                merged_model = model.prepare_for_merging(shared_compartment_name=self.shared_compartment_name)
                biomass_met_id = model.biomass_met.id
                biomass_met = merged_model.metabolites.get_by_id(biomass_met_id)
                biomass_mets[model.name] = biomass_met

                if self.fraction_reaction_flag:
                    rxn = cobra.Reaction(f"{model.name}_to_community_biomass")
                    rxn.add_metabolites({biomass_met: -1})
                    merged_model.add_reactions([rxn])
                    self.create_fraction_reaction(merged_model, member_name=model.name)

            else:
                extended_model = model.prepare_for_merging(shared_compartment_name=self.shared_compartment_name)

                unbalanced_metabolites = check_mass_balance_of_metabolites_with_identical_id(extended_model,
                                                                                             merged_model)
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

                biomass_met_id = model.biomass_met.id

                if self.fraction_reaction_flag:
                    biomass_met = extended_model.metabolites.get_by_id(biomass_met_id)
                    biomass_mets[model.name] = biomass_met
                    rxn = cobra.Reaction(f"{model.name}_to_community_biomass")
                    rxn.add_metabolites({biomass_met: -1})
                    extended_model.add_reactions([rxn])
                    self.create_fraction_reaction(extended_model, member_name=model.name)

                merged_model.merge(extended_model)
                biomass_mets[model.name] = merged_model.metabolites.get_by_id(biomass_met_id)

        if self.fraction_reaction_flag:
            self.fixed_growth_rate_flag = True
            self.fixed_abundance_flag = False
            self.merge_fraction_reactions(merged_model)
            self._add_fixed_abundance_reaction(merged_model)

        # old implementation of community biomass reactions
        if not self.fraction_reaction_flag:
            biomass_met = cobra.Metabolite(f"cpd11416_{self.shared_compartment_name}", name='Community Biomass',
                                           compartment=self.shared_compartment_name)
            merged_model.add_metabolites([biomass_met])
            biomass_rxn = cobra.Reaction("community_biomass")
            biomass_rxn.name = "Community Biomass Reaction"
            biomass_rxn.lower_bound = 0.
            biomass_rxn.upper_bound = 1000.
            rxn_mets = {}
            for model, met in biomass_mets.items():
                rxn_mets[met] = -1.
            rxn_mets[biomass_met] = 1.
            biomass_rxn.add_metabolites(rxn_mets)
            merged_model.add_reactions([biomass_rxn])

            cpd11416_exchg = merged_model.metabolites.get_by_id(f"cpd11416_{self.shared_compartment_name}")
            cpd11416_exchg_rxn = cobra.Reaction(f"EX_cpd11416_{self.shared_compartment_name}",
                                                name="community biomass exchange")
            cpd11416_exchg_rxn.lower_bound = 0.
            cpd11416_exchg_rxn.upper_bound = 1000.
            cpd11416_exchg_rxn.add_metabolites({cpd11416_exchg: -1})
            merged_model.add_reactions([cpd11416_exchg_rxn])

        # new implementation of community biomass reaction
        else:
            biomass_met = cobra.Metabolite(f"cpd11416_{self.shared_compartment_name}", name='Community Biomass',
                                           compartment=self.shared_compartment_name)
            biomass_rxn = cobra.Reaction("community_biomass")
            biomass_rxn.add_metabolites({biomass_met: -1})
            merged_model.add_reactions([biomass_rxn])
            # create additional reactions for each biomass reaction of a suborganism
            for member, met in biomass_mets.items():
                rxn = merged_model.reactions.get_by_id(f"{member}_to_community_biomass")
                rxn.add_metabolites({met: -1, biomass_met: 1}, combine=False)
            # set mu_c for the community, default = 1
            self.apply_fixed_growth_rate(self.mu_c, merged_model)

        merged_model.objective = "community_biomass"

        # Remove old biomass reactions
        old_biomass_exchange_rxns = []
        for model, met in biomass_mets.items():
            for rxn in met.reactions:
                if rxn in merged_model.exchanges:
                    old_biomass_exchange_rxns.append(rxn)
        merged_model.remove_reactions(old_biomass_exchange_rxns, remove_orphans=True)
        make_model_ids_sbml_conform(merged_model)
        merged_model.id = self.name

        self._unconstrained_model = merged_model

        if not self.is_mass_balanced():
            print(
                "WARNING: Not all reactions in the model are mass and charge balanced. To check which reactions are "
                "imbalanced, please run the get_unbalanced_reactions method of this CommunityModel object")

        return merged_model

    def create_fraction_reaction(self, model, member_name):
        fraction_reaction = cobra.Reaction(f"{member_name}_fraction_reaction")
        # create f_final metabolite
        f_final_met = cobra.Metabolite("f_final_met", name='Final Fraction Reaction Metabolite',
                                       compartment='fraction_reaction')
        fraction_reaction.add_metabolites({f_final_met: 1})
        # create biomass_metabolite for fraction reaction
        f_biomass_met = cobra.Metabolite(f'{member_name}_f_biomass_met', name=f'Fraction Biomass Metabolite of {member_name}',
                                         compartment='fraction_reaction')

        biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
        fraction_reaction.add_metabolites({f_biomass_met: 1})
        biomass_rxn.add_metabolites({f_biomass_met: -1})
        self._backup_metabolites[f"{member_name}_f_biomass_met"] = f_biomass_met
        # add fraction reaction to model
        model.add_reactions([fraction_reaction])
        # convert constraints of S.O.M to metabolites and add them to fraction reaction and constrained S.O.M reactions
        constraint_mets = self.convert_constraints_to_metabolites(model, member_name)
        self.add_sink_reactions_to_metabolites(model, constraint_mets)

    def convert_constraints_to_metabolites(self, model, member_name):
        # create empty dictionary for constrained metabolites
        constrained_mets = {}  # keys: metabolite, values: coefficent
        fraction_reaction_mets = {}
        fraction_reaction = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
        exchange_rxns = model.exchanges

        for reaction in model.reactions:
            if reaction == fraction_reaction or reaction in exchange_rxns:
                continue
            else:
                # create constraint metabolites
                met_lb = cobra.Metabolite(f'{reaction.id}_lb', name=f'{reaction.id} lower bound',
                                          compartment='fraction_reaction')
                met_ub = cobra.Metabolite(f'{reaction.id}_ub', name=f'{reaction.id} upper bound',
                                          compartment='fraction_reaction')
                # add constrained metabolites to the constrained_mets dictionary
                if reaction.lower_bound != 0:
                    coefficient = -self.max_flux if reaction.lower_bound < -self.max_flux else reaction.lower_bound
                    constrained_mets[met_lb] = coefficient
                    fraction_reaction_mets[met_lb] = -coefficient
                    reaction.add_metabolites({met_lb: 1})
                if reaction.upper_bound != 0:
                    coefficient = self.max_flux if reaction.upper_bound > self.max_flux else reaction.upper_bound
                    constrained_mets[met_ub] = coefficient
                    fraction_reaction_mets[met_ub] = coefficient
                    reaction.add_metabolites({met_ub: -1})

                # Relax reaction bounds
                if reaction.lower_bound < 0:
                    reaction.lower_bound = -self.max_flux
                else:
                    reaction.lower_bound = 0
                if reaction.upper_bound <= 0:
                    reaction.upper_bound = 0
                else:
                    reaction.upper_bound = self.max_flux

        # Add fraction metabolites to the fraction reaction
        fraction_reaction.add_metabolites(fraction_reaction_mets)
        return constrained_mets

    def change_reaction_bounds(self, reaction: str or cobra.Reaction, lower_bound, upper_bound):
        model = self.community_model

        if isinstance(reaction, str):
            reaction = model.reactions.get_by_id(reaction)

        # Is reaction part of a community member?
        member_name = self.get_member_name_of_reaction(reaction)

        if member_name is None:
            # No scaling to do
            reaction.bounds = (lower_bound, upper_bound)
            return

        # Add new fraction metabolites
        # create empty dictionary for constrained metabolites
        metabolites_needing_sink_reactions = []
        fraction_reaction_mets = {}
        fraction_reaction = model.reactions.get_by_id(f"{member_name}_fraction_reaction")

        # get or create constraint metabolites
        if model.metabolites.has_id(f'{reaction.id}_lb'):
            met_lb = model.metabolites.get_by_id(f'{reaction.id}_lb')
        else:
            met_lb = cobra.Metabolite(f'{reaction.id}_lb', name=f'{reaction.id} lower bound',
                                      compartment='fraction_reaction')
            metabolites_needing_sink_reactions.append(met_lb)

        if model.metabolites.has_id(f'{reaction.id}_ub'):
            met_ub = model.metabolites.get_by_id(f'{reaction.id}_ub')
        else:
            met_ub = cobra.Metabolite(f'{reaction.id}_ub', name=f'{reaction.id} upper bound',
                                      compartment='fraction_reaction')
            metabolites_needing_sink_reactions.append(met_ub)

        # Assign the constraint metabolites the reaction coefficients
        if lower_bound != 0:
            coefficient = -self.max_flux if lower_bound < -self.max_flux else lower_bound
            fraction_reaction_mets[met_lb] = -coefficient
            reaction.add_metabolites({met_lb: 1}, combine=False)
        else:
            reaction.add_metabolites({met_lb: 0}, combine=False)
            fraction_reaction_mets[met_lb] = 0

        if upper_bound != 0:
            coefficient = self.max_flux if upper_bound > self.max_flux else upper_bound
            fraction_reaction_mets[met_ub] = coefficient
            reaction.add_metabolites({met_ub: -1}, combine=False)
        else:
            fraction_reaction_mets[met_ub] = 0
            reaction.add_metabolites({met_ub: 0}, combine=False)

        # Relax reaction bounds
        if lower_bound < 0:
            reaction.lower_bound = -self.max_flux
        else:
            reaction.lower_bound = 0
        if upper_bound <= 0:
            reaction.upper_bound = 0
        else:
            reaction.upper_bound = self.max_flux

        # Add fraction metabolites to the fraction reaction
        fraction_reaction.add_metabolites(fraction_reaction_mets, combine=False)

        # Add sink reactions for fraction mets
        self.add_sink_reactions_to_metabolites(model, metabolites_needing_sink_reactions)

    def add_sink_reactions_to_metabolites(self, model, constraint_mets, lb=0., inplace=True):
        sink_max_flux = 1000000

        if sink_max_flux < 10*self.max_flux:
            sink_max_flux = 10*self.max_flux

        if not inplace:
            model = model.copy()

        for met in constraint_mets:
            model.add_boundary(met, type="sink", lb=lb, ub=sink_max_flux)

        model.repair()

        if inplace:
            return
        else:
            return model

    def merge_fraction_reactions(self, merged_model):
        # create f_final reaction
        # This ensures that the fractions sum up to 1
        f_final = cobra.Reaction("f_final", name="final fraction reaction")
        f_final.bounds = (1, 1)
        f_final_met = merged_model.metabolites.get_by_id("f_final_met")
        f_final.add_metabolites({f_final_met: -1})
        merged_model.add_reactions([f_final])

    def apply_fixed_growth_rate(self, flux, model=None):
        if not self.fixed_growth_rate_flag:
            print("Error: The model needs to be in fixed growth rate structure to set a fixed growth rate.")
            return
        self.mu_c = flux

        if model is None:
            model = self.community_model
        model.reactions.get_by_id("community_biomass").bounds = (flux, flux)

        for member_name in self.get_member_names():
            fraction_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            try:
                fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                fraction_rxn.add_metabolites({fraction_met: flux}, combine=False)
            except KeyError:
                fraction_met = self._backup_metabolites[f"{member_name}_f_biomass_met"]
                fraction_rxn.add_metabolites({fraction_met: flux}, combine=False)

        model.repair()

    def _add_fixed_abundance_reaction(self, model):
        # add an abundance reaction to the model for fixed abundance model structure (used later)
        abd_rxn = cobra.Reaction("abundance_reaction")
        abd_rxn.name = "Abundance Reaction"
        abd_rxn.bounds = (0., 0.)  # Not used in fixed growth

        abd_rxn_mets = {}
        for member_name in self.get_member_names():
            f_bio_met = model.metabolites.get_by_id(f'{member_name}_f_biomass_met')
            abd_rxn_mets[f_bio_met] = 1.

        abd_rxn.add_metabolites(abd_rxn_mets)

        model.add_reactions([abd_rxn])

    def convert_to_fixed_abundance(self, abundance_dict=None):
        """This function changes the model structure to fixed abundance, but variable growth rate. The model is left
        unchanged if it is already in fixed abundance structure."""
        if self.fixed_abundance_flag:
            print(f"Note: Model already has fixed abundance structure.")
            return

        model = self.community_model

        # Remove the f_bio metabolites from the fraction reactions
        for member_name in self.get_member_names():
            fraction_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
            fraction_rxn.add_metabolites({fraction_met: 0}, combine=False)

        # Activate the source reaction for coupled f_bio metabolites, constrained to the abundance
        model.reactions.get_by_id("abundance_reaction").bounds = (0., self.max_flux)

        # Relax the community biomass reaction constraints
        model.reactions.get_by_id("community_biomass").bounds = (0., self.max_flux)

        # Set the model structure flags correctly
        self.fixed_abundance_flag = True
        self.fixed_growth_rate_flag = False

        # Apply the abundance - if none was specified, use equal abundance
        if abundance_dict is None:
            if self._abundance_dict is None:
                abundance_dict = self.generate_equal_abundance_dict()
            else:
                abundance_dict = self._abundance_dict

        self.apply_fixed_abundance(abundance_dict)

        return

    def apply_fixed_abundance(self, abd_dict):
        """Applying fixed abundance to the model. This is only available if the model is in fixed abundance structure
        (check fixed_abundance_flag)."""
        if not self.fixed_abundance_flag:
            print("Error: the model is not in fixed abundance structure, but fixed abundance was tried to be applied. "
                  "Convert the model to fixed abundance structure first.")
            return

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

        # Apply the abundance as ratios of f_biomass metabolites
        model = self.community_model
        abd_rxn_mets = {}
        for member_name, fraction in abd_dict.items():
            try:
                f_bio_met = model.metabolites.get_by_id(f'{member_name}_f_biomass_met')
            except KeyError:
                f_bio_met = self._backup_metabolites[f"{member_name}_f_biomass_met"]
            f_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            f_rxn.bounds = (0., fraction)
            abd_rxn_mets[f_bio_met] = fraction

        abd_rxn = model.reactions.get_by_id("abundance_reaction")
        abd_rxn.add_metabolites(abd_rxn_mets, combine=False)

        self._abundance_dict = abd_dict

        model.repair()

        return

    def convert_to_fixed_growth_rate(self, mu_c=None):
        """This function changes the model structure to fixed growth rate, but variable abundance profile. The model
        is left unchanged if it is already in fixed abundance structure."""
        if self.fixed_growth_rate_flag:
            print(f"Note: Model already has fixed growth rate structure.")
            return

        model = self.community_model

        if mu_c is None:
            mu_c = self.mu_c

        # Deactivate the source reaction for coupled f_bio metabolites, constrained to the abundance
        model.reactions.get_by_id("abundance_reaction").bounds = (0., 0.)

        # Reset the fraction reaction bounds
        for member_name in self.get_member_names():
            f_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            f_rxn.bounds = (0., 1.)

        # Set the model structure flags correctly
        self.fixed_abundance_flag = False
        self.fixed_growth_rate_flag = True

        # Add the f_biomass metabolites to the fraction reactions and apply mu_c to the biomass reaction bounds
        self.apply_fixed_growth_rate(flux=mu_c)

        return

    def generate_equal_abundance_dict(self):
        abundances = {}
        names = self.get_member_names()
        for name in names:
            abundances[name] = 1. / len(names)
        return abundances

    def equal_abundance(self):
        if not self.fixed_abundance_flag:
            self.convert_to_fixed_abundance()
        abundances = self.generate_equal_abundance_dict()
        self.apply_fixed_abundance(abundances)

    def convert_to_model_without_fraction_metabolites(self):
        was_fixed_growth = False
        if self.fixed_growth_rate_flag:
            was_fixed_growth = True
            self.convert_to_fixed_abundance()

        model = self.community_model.copy()

        if was_fixed_growth:
            self.convert_to_fixed_growth_rate()

        reactions_to_remove = [model.reactions.get_by_id("f_final")]

        for reaction in model.reactions:
            if "fraction_reaction" in reaction.id:
                reactions_to_remove.append(reaction)
                for metabolite, coeff in reaction.metabolites.items():
                    if "_lb" == metabolite.id[-3:]:
                        rxn = model.reactions.get_by_id(metabolite.id[:-3])
                        rxn.lower_bound = -coeff
                        rxn.add_metabolites({metabolite: 0}, combine=False)
                        reaction.add_metabolites({metabolite: 0}, combine=False)
                        metabolite.remove_from_model(True)
                    elif "_ub" == metabolite.id[-3:]:
                        rxn = model.reactions.get_by_id(metabolite.id[:-3])
                        rxn.upper_bound = coeff
                        rxn.add_metabolites({metabolite: 0}, combine=False)
                        reaction.add_metabolites({metabolite: 0}, combine=False)
                        metabolite.remove_from_model(True)
                    elif "_f_biomass_met" in metabolite.id:
                        rxn = model.reactions.get_by_id(metabolite.id.split("_f_biomass_met")[0] + "_to_community_biomass")
                        rxn.add_metabolites({metabolite: 0}, combine=False)
                        reaction.add_metabolites({metabolite: 0}, combine=False)
                        metabolite.remove_from_model(True)

        for reaction in reactions_to_remove:
            reaction.remove_from_model(remove_orphans=True)

        return model

    def load_medium_from_file(self, file_path):
        # load the medium dictionary
        medium_dict = read_medium_from_file(file_path, comp=f"_{self.shared_compartment_name}")
        self.medium = medium_dict

    def apply_medium(self):
        test_if_medium_exists = self.medium
        medium_model = self.community_model
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
            solution_df.to_csv(file_path, sep="\t", header=True, index=False, float_format='%f')
        return solution_df

    def _run_fva_with_no_medium_for_loops(self, composition_agnostic=False, loopless=False):
        """This function is only used for finding loops. It converts the model to fixed growth rate of 0 if
        composition agnostic is wished. All changes are reverted."""
        fraction_of_optimum = 0.
        model = self.community_model
        model.medium = {}
        non_fraction_reactions = model.reactions.query(lambda x: "fraction_reaction" not in x.id and not (
                x.id[:3] == "SK_" and x.id[-3:] in ["_ub", "_lb"]))

        if composition_agnostic:
            if self.fixed_growth_rate_flag:
                mu_c = self.mu_c
                self.apply_fixed_growth_rate(0.)
                solution_df = find_loops_in_model(model)
                self.apply_fixed_growth_rate(mu_c)
            else:
                self.convert_to_fixed_growth_rate(mu_c=0.)
                solution_df = find_loops_in_model(model)
                self.convert_to_fixed_abundance()
        else:
            solution_df = find_loops_in_model(model)

        solution_df.insert(loc=0, column='reaction', value=list(solution_df.index))
        solution_df.columns = ["reaction_id", "min_flux", "max_flux"]

        return solution_df

    def run_fva(self, unconstrained=False, fraction_of_optimum=0.9, composition_agnostic=False, loopless=False, fva_mu_c=None):
        if unconstrained:
            model = self.unconstrained_model
        else:
            model = self.community_model

        if fva_mu_c is None and composition_agnostic:
            fva_mu_c = 0.
        elif fva_mu_c is not None:
            fraction_of_optimum = 1.

        reactions = model.reactions.query(lambda x: any([met.compartment == self.shared_compartment_name
                                                         for met in x.metabolites.keys()]))

        if composition_agnostic:
            if self.fixed_growth_rate_flag:
                mu_c = self.mu_c
                self.apply_fixed_growth_rate(0.)
                # Allow flux through the biomass reaction
                self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=self.max_flux)

                # Uncouple the biomass flux from the fraction reactions. This allows a model structure where member
                # organisms have their fluxes scaled by abundance, but their growth rate is not equal. This allows to
                # discover a superset of possible metabolite exchanges.
                f_bio_mets = {}
                for member_name in self.get_member_names():
                    biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                    fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                    f_bio_mets[member_name] = fraction_met
                    biomass_rxn.add_metabolites({fraction_met: 0}, combine=False)

                solution_df = cobra.flux_analysis.flux_variability_analysis(self.community_model,
                                                                            reactions,
                                                                            fraction_of_optimum=fraction_of_optimum,
                                                                            loopless=loopless)

                # Revert changes
                self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=0.)
                for member_name, fraction_met in f_bio_mets.items():
                    biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                    biomass_rxn.add_metabolites({fraction_met: -1}, combine=False)

                self.apply_fixed_growth_rate(mu_c)
            else:
                self.convert_to_fixed_growth_rate(mu_c=0.)

                # Uncouple the biomass flux from the fraction reactions. This allows a model structure where member
                # organisms have their fluxes scaled by abundance, but their growth rate is not equal. This allows to
                # discover a superset of possible metabolite exchanges.
                f_bio_mets = {}
                for member_name in self.get_member_names():
                    biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                    fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                    f_bio_mets[member_name] = fraction_met
                    biomass_rxn.add_metabolites({fraction_met: 0}, combine=False)

                solution_df = cobra.flux_analysis.flux_variability_analysis(self.community_model,
                                                                            reactions,
                                                                            fraction_of_optimum=fraction_of_optimum,
                                                                            loopless=loopless)

                # Revert changes
                self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=0.)
                for member_name, fraction_met in f_bio_mets.items():
                    biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                    biomass_rxn.add_metabolites({fraction_met: -1}, combine=False)

                self.convert_to_fixed_abundance()
        else:
            solution_df = cobra.flux_analysis.flux_variability_analysis(self.community_model, reactions,
                                                                        fraction_of_optimum=fraction_of_optimum,
                                                                        loopless=loopless)

        solution_df.insert(loc=0, column='reaction', value=list(solution_df.index))
        solution_df.columns = ["reaction_id", "min_flux", "max_flux"]

        return solution_df

    def fva_solution_flux_vector(self, file_path="", unconstrained=False, fraction_of_optimum=0.9):
        solution_df = self.run_fva(unconstrained=unconstrained, fraction_of_optimum=fraction_of_optimum)

        if len(file_path) > 0:
            print(f"Saving flux vector to {file_path}")
            solution_df.to_csv(file_path, sep="\t", header=True, index=False, float_format='%f')
        return solution_df

    def cross_feeding_metabolites_from_fba(self, unconstrained=False):
        if unconstrained:
            model = self.unconstrained_model
        else:
            model = self.community_model

        solution_df = self.run_fba(unconstrained=unconstrained)
        rows = []

        exchg_metabolites = model.metabolites.query(lambda x: x.compartment == self.shared_compartment_name)
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

    def cross_feeding_metabolites_from_fva(self, unconstrained=False, fraction_of_optimum=0.,
                                           composition_agnostic=False, fva_mu_c=None):
        if unconstrained:
            model = self.unconstrained_model
        else:
            model = self.community_model

        solution_df = self.run_fva(unconstrained=unconstrained, fraction_of_optimum=fraction_of_optimum,
                                   composition_agnostic=composition_agnostic, fva_mu_c=fva_mu_c)
        rows = []
        exchg_metabolites = model.metabolites.query(lambda x: x.compartment == self.shared_compartment_name)
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
            row_dict = {"metabolite_id": row["metabolite_id"], "metabolite_name": row["metabolite_name"],
                        "cross_feeding": row["cross_feeding"], "produced_by": [], "consumed_by": []}
            for member in member_names:
                if row[member + "_min_flux"] < 0.:
                    row_dict["consumed_by"].append(member)
                if row[member + "_max_flux"] > 0.:
                    row_dict["produced_by"].append(member)
            rows.append(row_dict)
        exchg_metabolite_df = pd.DataFrame(rows)

        return exchg_metabolite_df

    def potential_metabolite_exchanges(self, fba=False, fva_mu_c=None):
        if fba:
            exchange_df = self.cross_feeding_metabolites_from_fba(unconstrained=False)
        elif fva_mu_c is not None:
            exchange_df = self.cross_feeding_metabolites_from_fva(unconstrained=False, fraction_of_optimum=1.,
                                                                  fva_mu_c=fva_mu_c, composition_agnostic=False)
        else:
            exchange_df = self.cross_feeding_metabolites_from_fva(unconstrained=False, fraction_of_optimum=1.,
                                                                  fva_mu_c=fva_mu_c, composition_agnostic=True)
        return self.format_exchg_rxns(exchange_df)

    def report(self, verbose=True, max_reactions=5000):
        report_dict = {}
        model_structure = "fixed growth rate" if self.fixed_growth_rate_flag else "fixed abundance"
        num_metabolites = len(self.community_model.metabolites)
        num_f_metabolites = len(self.f_metabolites)
        num_model_metabolites = num_metabolites - num_f_metabolites
        num_reactions = len(self.community_model.reactions)
        num_f_reactions = len(self.f_reactions)
        num_model_reactions = num_reactions - num_f_reactions
        num_genes = len(self.community_model.genes)
        member_names = self.get_member_names()
        num_members = len(member_names)
        objective_expression = self.community_model.objective.expression
        objective_direction = self.community_model.objective.direction
        unbalanced_reactions = self.get_unbalanced_reactions()
        num_unbalanced_reactions = len(unbalanced_reactions)

        reactions_in_loops = "NaN"
        num_loop_reactions = "NaN"
        if num_model_reactions <= max_reactions:
            reactions_in_loops = self.get_loops()
            num_loop_reactions = len(reactions_in_loops)
        else:
            print(f"Note: The model has more than {max_reactions} reactions. Calculation of loops is skipped, as this "
                  f"would take some time. If needed, please run manually via .get_loops()")
        report_dict = {"community_name": self.name,
                       "model_structure": model_structure,
                       "num_metabolites": num_metabolites,
                       "num_f_metabolites": num_f_metabolites,
                       "num_model_metabolites": num_model_metabolites,
                       "num_reactions": num_reactions,
                       "num_f_reactions": num_f_reactions,
                       "num_model_reactions": num_model_reactions,
                       "num_genes": num_genes,
                       "member_names": member_names,
                       "num_members": num_members,
                       "objective_expression": objective_expression,
                       "objective_direction": objective_direction,
                       "unbalanced_reactions": unbalanced_reactions,
                       "num_unbalanced_reactions": num_unbalanced_reactions,
                       "reactions_in_loops": reactions_in_loops,
                       "num_loop_reactions": num_loop_reactions
                       }
        if verbose:
            print(f"Name: {self.name}")
            print("------------------")
            print("Model overview")
            print(f"Model structure: {model_structure}")
            print(f"# Metabolites: {num_metabolites}")
            print(f"# Constraint (f-) Metabolites: {num_f_metabolites}")
            print(f"# Model Metabolites: {num_model_metabolites}")
            print(f"# Reactions: {num_reactions}")
            print(f"# Constraint (f-) Reactions: {num_f_reactions}")
            print(f"# Model Reactions: {num_model_reactions}")
            print(f"# Genes: {num_genes}")
            print(f"# Members: {num_members}")
            print(f"Members:")
            for member in member_names:
                print(f"\t{member}")
            print(f"Objective in direction {objective_direction}:\n\t{objective_expression}")
            print("------------------")
            print("Model quality")
            print(f"# Reactions unbalanced: {num_unbalanced_reactions}")
            print(f"# Reactions able to carry flux without a medium: {num_loop_reactions}")

        return report_dict

    def save(self, file_path):
        """
        Save the community model object as a SBML file. This also includes the names of the community members and their abundance (if set).
        """
        # Generate a libsbml.model object
        cobra.io.write_sbml_model(self.community_model, filename=file_path)
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

        create_parameter_in_sbml_model(sbml_model, "fixed_abundance_flag", False,
                                       value=(1 if self.fixed_abundance_flag else 0))
        create_parameter_in_sbml_model(sbml_model, "fixed_growth_rate_flag", False,
                                       value=(1 if self.fixed_growth_rate_flag else 0))
        create_parameter_in_sbml_model(sbml_model, "mu_c", False, value=self.mu_c)
        create_parameter_in_sbml_model(sbml_model, "shared_compartment_id", True, value=self.shared_compartment_name,
                                       as_name=True)

        libsbml.writeSBMLToFile(sbml_doc, file_path)

        return

    @classmethod
    def load(cls, file_path):

        abundance_parameters = get_abundance_parameters_from_sbml_file(file_path)
        assert len(abundance_parameters) > 0

        flags_and_muc = get_flags_and_muc_from_sbml_file(file_path)
        assert len(flags_and_muc) == 4

        constructor_args = {}
        constructor_args["member_names"] = list(abundance_parameters.keys())
        if any([val is not None for val in abundance_parameters.values()]):
            constructor_args["abundance_profile"] = abundance_parameters
        constructor_args["unconstrained_model"] = cobra.io.read_sbml_model(file_path)

        constructor_args["fixed_abundance"] = flags_and_muc["fixed_abundance_flag"]
        constructor_args["fixed_growth_rate"] = flags_and_muc["fixed_growth_rate_flag"]

        name = constructor_args["unconstrained_model"].id
        return cls(name=name, mu_c=flags_and_muc["mu_c"], **constructor_args)


def doall(model_folder="", models=None, com_model=None, out_dir="", community_name="community_model",
          fixed_growth_rate=None, abundance="equal", medium=None,
          fba_solution_path=None, fva_solution_path=None, fva_solution_threshold=0.9, fba_interaction_path=None,
          fva_interaction_path=None, sbml_output_file=None, return_as_cobra_model=False,
          merge_via_annotation=None):
    com_model_obj = None
    if com_model is not None:
        # Load community model
        com_model_obj = CommunityModel.load(com_model)
    else:
        # Load single organism models
        # Either from folder or as list of file names or list of cobra models
        if model_folder != "":
            named_models = load_named_models_from_dir(model_folder)
        elif not isinstance(models, list) or len(models) == 0:
            raise ValueError(f"No models supplied to the doall function. Please supply either a path to the folder "
                             f"containing the models to the model_folder variable or a list of filepaths or cobra "
                             f"models"
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

    if fixed_growth_rate is None:
        # Apply abundance (either None, "equal", or an abundance dict)
        if abundance == "equal":
            com_model_obj.equal_abundance()
        elif isinstance(abundance, dict):
            name_conversion = com_model_obj.generate_member_name_conversion_dict()
            tmp_abundance = {}
            for name, fraction in abundance:
                tmp_abundance[name_conversion[name]] = fraction
            com_model_obj.convert_to_fixed_abundance()
            com_model_obj.apply_fixed_abundance(tmp_abundance)
        else:
            com_model_obj.convert_to_fixed_abundance()
    else:
        if fixed_growth_rate < 0.:
            print(f"Error: Specified growth rate is negative ({fixed_growth_rate}). PyCoMo will continue with a "
                  f"growth rate set to 0.")
            fixed_growth_rate = 0.
        com_model_obj.convert_to_fixed_growth_rate()
        com_model_obj.apply_fixed_growth_rate(fixed_growth_rate)

    # Apply medium
    if medium is not None:
        com_model_obj.load_medium_from_file(medium)
        com_model_obj.apply_medium()

    if sbml_output_file is not None:
        com_model_obj.save(os.path.join(out_dir, sbml_output_file))

    if fba_solution_path is not None:
        try:
            com_model_obj.fba_solution_flux_vector(file_path=os.path.join(out_dir, fba_solution_path))
        except cobra.exceptions.Infeasible:
            print(f"WARNING: FBA of community is infeasible. No FBA flux vector file was generated.")

    if fva_solution_path is not None:
        try:
            com_model_obj.fva_solution_flux_vector(file_path=os.path.join(out_dir, fva_solution_path),
                                                   fraction_of_optimum=fva_solution_threshold)
        except cobra.exceptions.Infeasible:
            print(f"WARNING: FVA of community is infeasible. No FVA flux vector file was generated.")

    if fva_interaction_path is not None:
        try:
            interaction_df = com_model_obj.potential_metabolite_exchanges(fba=False)
            print(f"Saving flux vector to {os.path.join(out_dir, fva_interaction_path)}")
            interaction_df.to_csv(file_path=os.path.join(out_dir, fva_interaction_path), sep="\t", header=True,
                                  index=False, float_format='%f')
        except cobra.exceptions.Infeasible:
            print(f"WARNING: FVA of community is infeasible. No FVA interaction file was generated.")

    if fba_interaction_path is not None:
        try:
            interaction_df = com_model_obj.potential_metabolite_exchanges(fba=True)
            print(f"Saving flux vector to {os.path.join(out_dir, fba_interaction_path)}")
            interaction_df.to_csv(file_path=os.path.join(out_dir, fba_interaction_path), sep="\t", header=True,
                                  index=False, float_format='%f')
        except cobra.exceptions.Infeasible:
            print(f"WARNING: FBA of community is infeasible. No FBA interaction file was generated.")

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
        args.abundance = read_abundance_from_file(args.abundance)

    if args.is_community:
        doall(com_model=args.input[0], community_name=args.name, out_dir=args.output_dir, abundance=args.abundance,
              medium=args.medium,
              fba_solution_path=args.fba_solution_path, fva_solution_path=args.fva_solution_path,
              fva_solution_threshold=args.fva_flux, fba_interaction_path=args.fba_interaction_path,
              fva_interaction_path=args.fva_interaction_path,
              sbml_output_file=args.sbml_output_path, return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation)

    elif len(args.input) == 1 and os.path.isdir(args.input[0]):
        doall(model_folder=args.input[0], community_name=args.name, out_dir=args.output_dir, abundance=args.abundance, medium=args.medium,
              fba_solution_path=args.fba_solution_path, fva_solution_path=args.fva_solution_path,
              fva_solution_threshold=args.fva_flux, fba_interaction_path=args.fba_interaction_path,
              fva_interaction_path=args.fva_interaction_path,
              sbml_output_file=args.sbml_output_path, return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation)
    else:
        doall(models=args.input, community_name=args.name, out_dir=args.output_dir, abundance=args.abundance, medium=args.medium,
              fba_solution_path=args.fba_solution_path, fva_solution_path=args.fva_solution_path,
              fva_solution_threshold=args.fva_flux, fba_interaction_path=args.fba_interaction_path,
              fva_interaction_path=args.fva_interaction_path,
              sbml_output_file=args.sbml_output_path, return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation)

    print("All done!")
    sys.exit(0)
