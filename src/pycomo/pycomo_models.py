#!/usr/bin/env python3

"""Authors: Michael Predl, Marianne Mießkes The pycomo module contains classes for single species and community
metabolic models. They extend the COBRApy classes by meta-information required for community model generation. The
community model can be used for simulation, transferred via the sbml format, setting abundances and generation of FBA
flux vector tables."""

# IMPORT SECTION
from typing import List, Union
from math import isnan

import cobra
import libsbml
import warnings
import logging

from .helper.logger import *
from .helper.utils import *
from .helper.cli import *
from .helper.multiprocess import *

logger = logging.getLogger("pycomo")


class SingleOrganismModel:
    """
    This class is used to bundle single organism metabolic models with their meta-information necessary for community
    metabolic model generation. The class also handles quality control and preprocessing for merging into a community
    model.

    :param model: The model as a COBRApy model object
    :type model: class:`cobra.Model`
    :param name: The name of the model
    :type name: str
    :param biomass_met: The biomass metabolite of the model
    :type biomass_met: class:`cobra.metabolite`
    :param biomass_reaction: The biomass reaction of the model
    :type biomass_reaction: class:`cobra.reaction`
    :param biomass_met_id: The ID of the biomass metabolite
    :type biomass_met_id: str
    :param prepared_model: The preprocessed model for merging
    :type prepared_model: class:`cobra.Model`
    :param shared_compartment_name: The name of the shared compartment to be used in the community metabolic model
    :type shared_compartment_name: str
    """
    model: cobra.Model
    name: str
    _original_name: str  # Used to make a conversion table at any point
    biomass_met: cobra.Metabolite = None
    biomass_reaction: cobra.Reaction = None
    biomass_met_id: str = None
    prepared_model: cobra.Model = None
    shared_compartment_name = None
    _name_via_annotation: str = None
    _exchange_met_name_conversion: dict = {}

    def __init__(self, model, name, biomass_met_id="", name_via_annotation=None, shared_compartment_name="medium"):
        """
        Constructor method

        :param model: The metabolic model of the organism as a COBRApy model object
        :param name: The name of the model
        :param biomass_met_id: The ID of the biomass metabolite
        :param name_via_annotation: The database to be used for matching boundary metabolites when merging into a
            community metabolic model. If None, matching of metabolites is done via metabolite IDs instead
        :param shared_compartment_name: The name of the shared compartment to be used in the community metabolic model
        """
        self.model = model.copy()
        self._original_name = name
        self.name = make_string_sbml_id_compatible(name, remove_ascii_escapes=True)
        if name != self.name:
            logger.warning(
                f"Warning: model name {name} is not compliant with sbml id standards and was changed to {self.name}")
        self.biomass_met_id = biomass_met_id
        self._name_via_annotation = name_via_annotation
        self.shared_compartment_name = shared_compartment_name

    def get_name_conversion(self):
        """
        Retrieve the original and current name of this model.

        :return: original name, current name
        """
        return self._original_name, self.name

    def set_name_via_annotation(self, name_via_annotation):
        """
        Set the name_via_annotation attribute of the model. If a model was already prepared for merging, the
        preprocessing is repeated with the newly set attribute.

        :param name_via_annotation: The database to be used for matching boundary metabolites when merging into a
            community metabolic model. If None, matching of metabolites is done via metabolite IDs instead
        """
        self._name_via_annotation = name_via_annotation
        if self.prepared_model is not None:
            self.prepare_for_merging()

    def rename_comp_in_met_id(self, model, old_comp, new_comp, inplace=True, change_name=False,
                              remove_comp_from_name=False):
        """
        Renames the compartments in metabolite IDs.

        :param model: Model to be changed
        :param old_comp: The compartment to be renamed
        :param new_comp: The new name of the compartment
        :param inplace: If True, change the input model, else a copy is created and changed
        :param change_name: If True, change the metabolite name in addition to the metabolite ID
        :param remove_comp_from_name: If True, remove the compartment from the metabolite name (if present)
        :return: Model with renamed compartments in metabolite IDs
        """
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
        """
        Renames the compartments in reaction IDs.

        :param model: Model to be changed
        :param old_comp: The compartment to be renamed
        :param new_comp: The new name of the compartment
        :param inplace: If True, change the input model, else a copy is created and changed
        :param change_name: If True, change the reaction name in addition to the metabolite ID
        :return: Model with renamed compartments in reaction IDs
        """
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
        """
        Rename compartments of a model

        :param model: Model to be changed
        :param rename: A dictionary containing old compartments as keys and the new compartment names as values
        :return: The model with renamed compartments
        """
        old_compartments = model.compartments
        for comp in rename.keys():
            if comp not in old_compartments:
                raise KeyError(f"Compartment {comp} not in model compartments")

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
        """
        This method adds exchange reactions for all metabolites in a specified compartment. The method is used during
        preprocessing for model merging, to generate exchange reactions for all metabolites in the new compartment
        shared between community members.

        :param model: Model to be changed
        :param comp: The compartment ID where exchange metabolites should be added (i.e. the new, shared compartment)
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: None if the inplace flag is set, otherwise the model with added exchange reactions
        """
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
        """
        Adds exchange reactions to a list of metabolites.

        :param model: Model to be changed
        :param mets: A list of metabolites where exchange reactions should be added
        :param lb: The lower bound for the exchange reactions
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: None if the inplace flag is set, otherwise the model with added exchange reactions
        """
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
        """
        A function for adding an exchange reaction. In difference to the COBRApy functionality, this function can also
        set exchange reactions in compartments other than the exchange compartment.

        :param model: Model to be changed
        :param metabolite: The metabolite where an exchange reaction should be added
        :param reaction_id: An ID for the new reaction ID
        :param lb: The lower bound for the exchange reaction
        :param ub: The upper bound for the exchange reaction
        :param sbo_term: The SBO term to be used for the reaction
        :return: The added exchange reaction
        """
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
        """
        This method finds metabolites that do not have exchange reactions. This is useful for checking whether all
        metabolites that are meant to be boundary metabolites actually have an exchange reaction connected.

        :param model: Model to be checked
        :param exchg_comp: The compartment to be searched for metabolites without exchange reactions
        :return: A list of metabolites without exchange reactions
        """
        if len(exchg_comp) == 0:
            exchg_comp = cobra.medium.find_external_compartment(model)
        comp_mets = model.metabolites.query(lambda x: x.compartment == exchg_comp)
        exchg_mets = [list(rxn.metabolites)[0] for rxn in model.exchanges]
        mets_without_exchg = list((set(comp_mets) - set(exchg_mets)) - {self.biomass_met})
        logger.debug(f"Metabolites in exchange compartment without exchange reactions: {mets_without_exchg}")
        return mets_without_exchg

    def convert_exchange_to_transfer_reaction(self, model, old_comp, inplace=True, max_flux=1000.):
        """
        When preprocessing the model for merging into a community metabolic model, a new shared compartment is added.
        This new compartment will be the compartment for boundary metabolites, containing all metabolites with exchange
        reactions. The previous boundary metabolites should then have their exchange reactions converted into transfer
        reactions, which transfer the metabolites into the new, shared compartment.

        :param model: The model to be changed
        :param old_comp: The ID of the old exchange compartment
        :param inplace: If True, the input model will be changed, else a copy is made
        :param max_flux: The maximum allowed flux in the model (used to open the constraints of the transfer reactions)
        :return: None if the inplace flag is set, otherwise the model with changed reactions
        """
        if not inplace:
            model = model.copy()

        for rxn in model.exchanges:
            rxn_string = rxn.id
            if old_comp in rxn_string and old_comp == rxn_string[-len(old_comp):]:
                # Replace the exchange reaction term with TF_
                if "EX_" in rxn.id:
                    rxn.id = rxn.id.replace("EX_", "TF_")
                # Remove the SBO term for exchange reaction
                if "sbo" in rxn.annotation and "SBO:0000627" in rxn.annotation["sbo"]:
                    rxn.annotation.pop("sbo")
                out_met_rxn = rxn.copy()
                new_met_stoich = {}
                for met, stoich in out_met_rxn.metabolites.items():
                    new_met_stoich[model.metabolites.get_by_id(self._exchange_met_name_conversion[met.id])] = -stoich
                rxn.add_metabolites(new_met_stoich)
                if "Exchange" in rxn.name:
                    rxn.name = rxn.name.replace("Exchange", "Transfer")
                rxn.bounds = (-max_flux, max_flux)

        model.repair()

        if inplace:
            return
        else:
            return model

    def add_boundary_metabolites_to_exchange_compartment(self, model, new_comp=None, old_comp="", inplace=True):
        """
        This function adds boundary metabolites to the new exchange compartment for every metabolite in the old external
         compartment

        :param model: Model to be changed
        :param new_comp: ID of the new compartment
        :param old_comp: ID of the old external compartment
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: None if the inplace flag is set, otherwise the updated model
        """
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
                        raise TypeError(
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
        """
        This method checks all metabolite IDs and reaction IDs for a suffix containing the compartment they are in.
        Should the name of the compartment not be part of the ID, the suffix is added, as it is necessary for some
        operations of this class. The reaction ID will only be changed to contain a compartment suffix if all
        metabolites that are part of the reaction are found in the same compartment.

        :param model: The model to be changed
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: The updated model
        """
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

    def add_exchange_compartment(self, model, exchg_comp_name=None, add_missing_transfers=False, inplace=True,
                                 max_flux=1000.):
        """
        Add a new exchange compartment. This will also copy all metabolites in the current external compartment to the
        exchange compartment, establish transfer reactions between the two compartments for all metabolites and add
        new exchange reactions for all copied metabolites.

        :param model: Model to be changed
        :param exchg_comp_name: Name of the new exchange compartment
        :param add_missing_transfers: If True, add exchange reactions and transfer reactions for all metabolites in
            the old external compartment that did not have any exchange reactions
        :param inplace: If True, the input model will be changed, else a copy is made
        :param max_flux: Maximum allowed flux
        :return: Updated model
        """
        if exchg_comp_name is None:
            exchg_comp_name = self.shared_compartment_name

        if not inplace:
            model = model.copy()
        old_exc_comp = cobra.medium.find_external_compartment(model)
        # Add metabolites
        self.add_boundary_metabolites_to_exchange_compartment(model, new_comp=exchg_comp_name, inplace=True)
        # Add transfer reactions
        if add_missing_transfers:
            mets_without_exchg = self.find_metabolites_without_exchange_rxn(model)
            self.add_exchange_reactions_to_metabolites(model, mets_without_exchg, inplace=True)
        self.convert_exchange_to_transfer_reaction(model, old_exc_comp, inplace=True, max_flux=max_flux)
        # Add exchange reactions
        self.add_exchange_reactions_to_compartment(model, exchg_comp_name, inplace=True)

        return model

    def prefix_metabolite_names(self, model, prefix, exclude_compartment="", inplace=True):
        """
        Adds a prefix to all metabolite IDs, that do not already have this prefix.

        :param model: Model to be changed
        :param prefix: The prefix to be added to the metabolite IDs
        :param exclude_compartment: Metabolites in this compartment are not changed
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: Updated model
        """
        if not inplace:
            model = model.copy()  # Don't want to change the original

        for metabolite in model.metabolites:
            if not metabolite.compartment == exclude_compartment:
                if prefix != metabolite.id[:len(prefix)]:
                    metabolite.id = f"{prefix}{metabolite.id}"

        model.repair()

        return model

    def prefix_reaction_names(self, model, prefix, exclude_compartment="", inplace=True):
        """
        Adds a prefix to all reaction IDs, that do not already have this prefix.

        :param model: Model to be changed
        :param prefix: The prefix to be added to the reaction IDs
        :param exclude_compartment: Reactions with this compartment suffix in their ID are not changed
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: Updated model
        """
        if not inplace:
            model = model.copy()  # Don't want to change the original

        for reaction in model.reactions:
            if len(exclude_compartment) > 0 and reaction.id[-len(exclude_compartment):] != exclude_compartment:
                if prefix != reaction.id[:len(prefix)]:
                    reaction.id = f"{prefix}{reaction.id}"

        model.repair()

        return model

    def prefix_gene_names(self, model, prefix, inplace=True):
        """
        Adds a prefix to all gene IDs, that do not already have this prefix. Also writes the original gene ID as the
        gene name, if it is not yet set.

        :param model: Model to be changed
        :param prefix: The prefix to be added to the gene IDs
        :param inplace: If True, the input model will be changed, else a copy is made
        :return: Updated model
        """
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
        """
        Removes all exchange reactions of the biomass metabolite. This is used to ensure that the biomass metabolite
        will only be consumed by the community biomass reaction.

        :param model: Model to be changed
        :param remove_all_consuming_rxns: If True, remove all reactions consuming the biomass metabolite as well
        :return: None
        """
        for reaction in self.biomass_met.reactions:
            if reaction == self.biomass_rxn:
                continue
            if reaction in model.exchanges:
                reaction.remove_from_model()
            elif remove_all_consuming_rxns and self.biomass_met in reaction.reactants:
                reaction.remove_from_model(remove_orphans=True)
        return

    def handle_boundary_biomass_reaction(self):
        """
        Checks if identified biomass reaction is a boundary reaction. If so, replace the boundary prefix from the
        reaction ID (EX, SK, DM). Also removes the SBO term associated with the boundary reaction. In case only a
        single reactant is present (which is expected), the reactant is moved from the boundary compartment into the
        'bio' compartment, to avoid further boundary reactions being added to it.
        """
        if self.biomass_rxn is None:
            logger.info("No biomass reaction identified, skipping handling of boundary biomass reactions")
            return
        if self.model.reactions.get_by_id(self.biomass_rxn.id) in self.model.boundary:
            logger.info(f"Biomass reaction {self.biomass_rxn.id} is a boundary reaction.")
            # Remove the SBO term for exchange reaction
            if "sbo" in self.biomass_rxn.annotation and "SBO:0000627" in self.biomass_rxn.annotation["sbo"]:
                self.biomass_rxn.annotation.pop("sbo")
            biomass_reactants = self.biomass_rxn.reactants
            if len(biomass_reactants) == 1:
                # Move biomass metabolite out of boundary compartment
                logger.info(f"Moving reactant {biomass_reactants[0]} out of the boundary compartment into compartment "
                            f"'bio', to avoid further boundary reactions being added to it.")
                biomass_reactants[0].compartment = "bio"
            elif len(biomass_reactants) == 0:
                logger.warning(f"Biomass reaction {self.biomass_rxn.id} has no reactants!")
            elif len(biomass_reactants) > 1:
                logger.warning(f"Biomass reaction {self.biomass_rxn.id} is a boundary reaction with more than 1 "
                               f"reactant!")
            # Rename biomass reaction ID
            if "EX_" in self.biomass_rxn.id:
                self.biomass_rxn.id = self.biomass_rxn.id.replace("EX_", "bio_")
            elif "SK_" in self.biomass_rxn.id:
                self.biomass_rxn.id = self.biomass_rxn.id.replace("SK_", "bio_")
            elif "DM_" in self.biomass_rxn.id:
                self.biomass_rxn.id = self.biomass_rxn.id.replace("DM_", "bio_")

    def prepare_for_merging(self, shared_compartment_name=None, max_flux=1000.):
        """
        Prepares the model for merging into a community metabolic model. The generated model is SBML conform and all
        genes, reactions, compartments and metabolites contain the information that they belong to this model (which is
        important for traceability after merging into a community metabolic model). The procedure is as follows:
        - Get the ID of the biomass metabolite, if not already specified
        - Remove the biomass exchange reaction
        - Ensure SBML conform IDs in the whole model
        - Add the compartment name as suffix for all metabolites and reactions
        - Add the new shared exchange compartment and populate it with the boundary metabolites
        - Prefix all compartments, genes, reactions and metabolites with the name of the model

        :param shared_compartment_name: The ID of the new shared exchange compartment
        :param max_flux: Maximum allowed flux
        :return: The model prepared for merging
        """
        if shared_compartment_name is not None:
            self.shared_compartment_name = shared_compartment_name
        self.prepared_model = self.model.copy()
        biomass_met, biomass_rxn = get_model_biomass_compound(self.prepared_model,
                                                              self.shared_compartment_name,
                                                              generate_if_none=True,
                                                              return_biomass_rxn=True)
        self.biomass_met = biomass_met
        self.biomass_rxn = biomass_rxn

        # Handle case where the biomass reaction is a boundary reaction
        self.handle_boundary_biomass_reaction()

        self.remove_biomass_exchange_rxn(self.prepared_model)

        # Remove ascii escape characters from sbml ids, as they are not compatible
        make_model_ids_sbml_conform(self.prepared_model)

        # Check that compartment names are in metabolite and reaction ids
        self.ensure_compartment_suffix(self.prepared_model)

        rename = {}
        for comp in self.prepared_model.compartments:
            rename[comp] = self.name + "_" + comp
        self.rename_compartment(self.prepared_model, rename)
        self.add_exchange_compartment(self.prepared_model, add_missing_transfers=True, max_flux=max_flux)
        self.prefix_metabolite_names(self.prepared_model, self.name + "_",
                                     exclude_compartment=self.shared_compartment_name)
        self.prefix_reaction_names(self.prepared_model, self.name + "_",
                                   exclude_compartment=self.shared_compartment_name)
        self.prefix_gene_names(self.prepared_model, self.name + "_")
        return self.prepared_model


class CommunityModel:
    """
    This class contains the community metabolic model and its meta information. It is used for generating the community
    metabolic model from its members models. It also provides functionality to analyse community metabolic models, such
    as detection of thermodynamically infeasible cycles and calculation of all possible exchanged metabolites.
    The community metabolic model can switch between two states: The fixed abundance state and the fixed growth rate
    state. These states are important for ensuring the model is in steady-state. Switching between the two states is
    possible at any point.

    :param model: The community metabolic model
    :param medium: The medium of the model, defined as a dictionary with exchange reaction IDs as keys and the maximum
        flux of the respective metabolite as value.
    :param f_metabolites: A list of f_metabolites (dummy metabolites controlling the reaction bounds)
    :param f_reactions: A list of f_reactions (dummy reactions controlling the reaction bounds)
    :param member_models: A list of the models of the community members (SingleOrganismModel class)
    :param name: The name of the community metabolic model
    :param medium_flag: A flag whether a medium has been applied to the model
    :param mu_c: The community growth rate
    :param fixed_abundance_flag: A flag, whether the community metabolic model is in fixed abundance state
    :param fixed_growth_rate_flag: A flag, whether the community metabolic model is in fixed growth rate state
    :param max_flux: The maximum flux for all reactions in the community metabolic model
    :param shared_compartment_name: The name of the compartment for sharing metabolites and the medium
    """
    member_models: List[SingleOrganismModel]
    name: str
    medium_flag: bool = False
    mu_c: float = 1.
    fixed_abundance_flag: bool = False
    fixed_growth_rate_flag: bool = False
    max_flux: float = 1000.
    shared_compartment_name: str = None
    no_annotation_overlap: list = None
    _dummy_metabolite_scaling_factor = 0.01
    _f_metabolites: list = None
    _f_reactions: list = None
    _transfer_reactions: list = None
    _transport_reactions: list = None
    _model: cobra.Model = None
    _medium: dict = None
    _merge_via_annotation: str = None
    _abundance_dict: dict = None
    _member_names: list = None
    _backup_metabolites: dict = {}

    def __init__(self, models=None, name="", merge_via_annotation=None, mu_c=1., max_flux=1000.,
                 shared_compartment_name="medium", **kwargs):
        """
        Constructor method.

        :param models: A list of models of the community members as SingleOrganismModel objects
        :param name: The name of the community metabolic model
        :param merge_via_annotation: The database to be used for matching boundary metabolites when merging into a
            community metabolic model. If None, matching of metabolites is done via metabolite IDs instead
        :param mu_c: The community growth rate to be set
        :param max_flux: The maximum flux for all reactions in the community metabolic model
        :param shared_compartment_name: The name of the compartment for sharing metabolites and the medium
        """
        self.member_models = models
        self.mu_c = mu_c
        self.shared_compartment_name = shared_compartment_name
        self.no_annotation_overlap = []

        if models is not None:
            model_names = [model.name for model in self.member_models]
        elif "member_names" in kwargs.keys():  # Construction from saved file
            model_names = kwargs["member_names"]
        else:
            raise ValueError("No models provided to CommunityModel object!")

        if not list_contains_unique_strings(model_names):
            raise ValueError(f"Model names contain duplicates!")
        if list_of_strings_is_self_contained(model_names):
            raise ValueError(f"Some model names are contained in others!")

        self._member_names = model_names

        if max_flux > 0.:
            self.max_flux = max_flux
        else:
            logger.warning(
                f"Warning: maximum flux value is not greater than 0 ({max_flux}). Using default value of 1000.0 "
                f"instead.")
            self.max_flux = 1000.

        if len(name) == 0:
            name = "Community_Model"
            logger.warning(f"Warning: No name was given for the CommunityModel. Name is set to {name}")

        self.name = make_string_sbml_id_compatible(name)
        if name != self.name:
            logger.warning(
                f"Warning: model name {name} is not compliant with sbml id standards and was changed to {self.name}")
        if merge_via_annotation is not None:
            self._merge_via_annotation = merge_via_annotation
            for model in self.member_models:
                model.set_name_via_annotation(merge_via_annotation)

        if "fixed_abundance" in kwargs.keys():
            self.fixed_abundance_flag = kwargs["fixed_abundance"]

        if "fixed_growth_rate" in kwargs.keys():
            self.fixed_growth_rate_flag = kwargs["fixed_growth_rate"]

        if "model" in kwargs.keys():
            self._model = kwargs["model"]
            for member in self._member_names:
                try:
                    met = self._model.metabolites.get_by_id(f"{member}_f_biomass_met")
                except KeyError:
                    met = cobra.Metabolite(f'{member}_f_biomass_met', name=f'Fraction Biomass Metabolite of {member}',
                                           compartment='fraction_reaction')
                self._backup_metabolites[f"{member}_f_biomass_met"] = met

        if "abundance_profile" in kwargs.keys():
            self._abundance_dict = kwargs["abundance_profile"]

        if "shared_compartment_name" in kwargs.keys():
            self.shared_compartment_name = kwargs["shared_compartment_name"]

    @property
    def model(self):
        """
        The community metabolic model (COBRApy model object)

        :return: The community metabolic model (COBRApy model object)
        """
        return self._model

    @model.getter
    def model(self):
        """
        Getter for the community metabolic model (COBRApy model object). If no model is present, it will be generated
        from the community member models

        :return: The community metabolic model (COBRApy model object)
        """
        if self._model is None:
            self.no_annotation_overlap = []
            logger.info(f"No community model generated yet. Generating now:")
            self.generate_community_model()
            logger.info(f"Generated community model.")
            self.fixed_abundance_flag = False
            self.medium_flag = False
        return self._model

    @property
    def f_metabolites(self):
        """
        A list of f_metabolites (dummy metabolites controlling the reaction bounds)

        :return: A list of f_metabolites (dummy metabolites controlling the reaction bounds)
        """
        return self._f_metabolites

    @f_metabolites.getter
    def f_metabolites(self):
        """
        Getter function for the list of f_metabolites (dummy metabolites controlling the reaction bounds). Finds all
        metabolites in the compartment for dummy metabolites (fraction_reaction)

        :return: A list of f_metabolites (dummy metabolites controlling the reaction bounds)
        """
        self._f_metabolites = self.model.metabolites.query(lambda x: x.compartment == "fraction_reaction")
        if self._f_metabolites is None:
            self._f_metabolites = []
        return self._f_metabolites

    @property
    def f_reactions(self):
        """
        A list of f_reactions (dummy reactions controlling the reaction bounds)

        :return: A list of f_reactions (dummy reactions controlling the reaction bounds)
        """
        return self._f_reactions

    @f_reactions.getter
    def f_reactions(self):
        """
        Getter function for the list of f_reactions (dummy reactions controlling the reaction bounds)

        :return: A list of f_reactions (dummy reactions controlling the reaction bounds)
        """
        self._f_reactions = self.model.reactions.query(
            lambda x: (x.id[:3] == "SK_" and x.id[-3:] in {"_lb", "_ub"}) or "_fraction_reaction" in x.id)
        if self._f_reactions is None:
            self._f_reactions = []
        return self._f_reactions

    @property
    def transport_reactions(self):
        """
        A list of f_reactions (dummy reactions controlling the reaction bounds)

        :return: A list of f_reactions (dummy reactions controlling the reaction bounds)
        """
        return self._transfer_reactions

    @transport_reactions.getter
    def transport_reactions(self):
        """
        Getter function for the list of f_reactions (dummy reactions controlling the reaction bounds)

        :return: A list of f_reactions (dummy reactions controlling the reaction bounds)
        """
        self._transport_reactions = self.model.reactions.query(
            lambda x: self.is_transporter(x))
        if self._transport_reactions is None:
            self._transport_reactions = []
        return self._transport_reactions

    @property
    def transfer_reactions(self):
        """
        A list of f_reactions (dummy reactions controlling the reaction bounds)

        :return: A list of f_reactions (dummy reactions controlling the reaction bounds)
        """
        return self._transfer_reactions

    @transfer_reactions.getter
    def transfer_reactions(self):
        """
        Getter function for the list of f_reactions (dummy reactions controlling the reaction bounds)

        :return: A list of f_reactions (dummy reactions controlling the reaction bounds)
        """
        f_reactions = self.f_reactions
        transport_rxns = self.transport_reactions
        self._transfer_reactions = self.model.reactions.query(
            lambda x: x not in f_reactions
                      and x not in transport_rxns
                      and len(x.compartments) == 3
                      and "fraction_reaction" in x.compartments
                      and self.shared_compartment_name in x.compartments)
        if self._transfer_reactions is None:
            self._transfer_reactions = []
        return self._transfer_reactions

    @property
    def medium(self):
        """
        The medium of the community metabolic model

        :return: A dictionary of reaction IDs as keys and their maximum influx as values
        """
        return self._medium

    @medium.getter
    def medium(self):
        """
        Getter function for the medium of the community metabolic model.

        :raises ValueError: If no medium has been set so far, this error is raised
        :return: A dictionary of reaction IDs as keys and their maximum influx as values
        """
        if self._medium is None:
            raise ValueError("Error: No medium set for this community model.\nPlease set the medium with "
                             ".load_medium_from_file('/path/to/medium_file.csv')")
        return self._medium

    @medium.setter
    def medium(self, medium_dict):
        """
        Setter function for the medium of the community metabolic model.

        :raises ValueError: If not all keys of the medium dictionary are strings, or not all values are floats,
            this error is raised.
        :param medium_dict: The medium dictionary with exchange reaction IDs as keys and the maximum
            flux of the respective metabolite as value.
        """
        # Check that dataframe has the correct format
        if not all([isinstance(key, str) for key in medium_dict.keys()]):
            raise TypeError("Medium keys must be strings!")
        if not all([isinstance(value, float) for value in medium_dict.values()]):
            raise TypeError("Medium values must be floats!")
        self._medium = medium_dict
        self.apply_medium()

    def is_transporter(self, rxn):
        if rxn in self.f_reactions:
            # Fraction reactions are not transporters
            return False
        met_comps = [met.compartment for met in rxn.metabolites.keys() if met.compartment != "fraction_reaction"]
        if self.shared_compartment_name in met_comps:
            # Transporters operate within a single organism.
            # Exchange of metabolites between the shared compartment and the members is handled by transfer reactions.
            return False
        if len(set(met_comps)) < 2:
            # Transporters need to operate within at least 2 compartments
            return False
        for n in self.get_member_names():
            # Transporters need to operate within compartments of
            # the same member (excluding the fraction_reaction compartment)
            if n in rxn.id and all([n in m for m in met_comps]):
                return True
        return False

    def summary(self, suppress_f_metabolites=True, catch_infeasible=False):
        """
        Calls and returns the summary method of community metabolic model. Dummy metabolites and reactions can be
        excluded from the flux report by setting the suppress_f_metabolites flag.

        :param suppress_f_metabolites: If true, excludes dummy metabolites and reactions from the flux report
        :return: The COBRApy summary object of the community metabolic model
        """
        try:

            summary = self.model.summary()

            if suppress_f_metabolites:
                model = self.model
                new_secretion_flux_rows = []
                old_secretion_flux = summary.secretion_flux
                for idx, row in old_secretion_flux.iterrows():
                    if model.metabolites.get_by_id(row["metabolite"]).compartment != "fraction_reaction":
                        new_secretion_flux_rows.append(row)

                new_secretion_flux = pd.DataFrame(new_secretion_flux_rows)
                summary.secretion_flux = new_secretion_flux

            return summary
        except cobra.exceptions.Infeasible as e:
            if catch_infeasible:
                return "Infeasible!"
            else:
                raise e

    def generate_member_name_conversion_dict(self):
        """
        Creates a dictionary matching old and current names of each of the community members

        :return: Dictionary with old names as keys and current names as values
        """
        conversion_dict = {}
        if self.member_models is not None:
            for member in self.member_models:
                old_name, new_name = member.get_name_conversion()
                conversion_dict[old_name] = new_name
        else:
            logger.warning("Warning: There are no member models in the community model object.")
            for member_name in self.get_member_names():
                conversion_dict[member_name] = member_name
        return conversion_dict

    def get_member_names(self):
        """
        Get the names of all community members

        :return: A list of all community member names
        """
        if self.member_models is not None:
            member_names = [member.name for member in self.member_models]
        else:
            member_names = self._member_names
        return member_names

    def get_unbalanced_reactions(self):
        """
        Checks all functions in the community metabolic model for mass and charge balance. The check itself is
        performed by the function check_mass_balance of COBRApy.

        :return: A list of unbalanced reactions
        """
        try:
            unbalanced_reactions = check_mass_balance_fomula_safe(self.model)
        except TypeError:
            # This TypeError can come from multiple sbo terms being present in reaction annotations
            with self.model.copy() as single_reaction_sbo_model:
                for rxn in single_reaction_sbo_model.reactions:
                    if isinstance(rxn.annotation.get("sbo"), list):
                        rxn.annotation["sbo"] = rxn.annotation.get("sbo")[0]
                unbalanced_reactions = check_mass_balance_fomula_safe(single_reaction_sbo_model)
        return unbalanced_reactions

    def is_mass_balanced(self):
        """
        Checks if all reactions in the community metabolic model are mass and charge balanced. Returns True if mass and
        charge balance is achieved, else False.

        :return: True if mass and charge balance is achieved, else False
        """
        return not bool(self.get_unbalanced_reactions())

    def get_loops(self, processes=None):
        """
        This is a function to find closed loops that can sustain flux without any input or output. Such loops are
        thermodynamically infeasible and biologically nonsensical. Users should be aware of their presence and
        either remove them or check any model solutions for the presence of these cycles.

        :param processes: The number of processes to use
        :return: A DataFrame of reactions that carry flux without any metabolite input or output in the model
        """
        try:
            original_medium = self.medium
        except ValueError:
            original_medium = self.model.medium
        no_medium = {}
        self.model.medium = no_medium

        with self.model:
            solution_df = find_loops_in_model(self.convert_to_model_without_fraction_metabolites(), processes=processes)

        self.medium = original_medium
        self.apply_medium()

        return solution_df[
            (~ solution_df["min_flux"].apply(close_to_zero)) | (~ solution_df["max_flux"].apply(close_to_zero))]

    def get_member_name_of_reaction(self, reaction):
        """
        This function will return the name of the member the reaction belongs to by extracting this information from its
        ID.

        :param reaction: The reaction or reaction ID whose community member should be found
        :return: The name of the community member the reaction belongs to
        """
        if isinstance(reaction, str):
            reaction = self.model.reactions.get_by_id(reaction)

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

        :param metabolite: The metabolite or metabolite ID whose community member should be found
        :return: The name of the community member the metabolite belongs to
        """
        if isinstance(metabolite, str):
            metabolite = self.model.metabolites.get_by_id(metabolite)

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

        :param compartment: The compartment ID whose community member should be found
        :return: The name of the community member the compartment belongs to
        """
        member_name = None

        for name in self.get_member_names():
            if name == compartment[:len(name)]:
                member_name = name
                break

        return member_name

    def generate_community_model(self):
        """
        This method generates a community metabolic model by merging the models of its community members. The resulting
        community metabolic model is in fixed growth state. The procedure is as follows:
        - Each of the community member models is preprocessed for merging (see SingleOrganismModel class for details)
        - The preprocessed models are converted into equivalent, bound-free form
        - The preprocessed, bound-free models are merged into a community metabolic model
        - The community biomass reaction is created and added
        - A fixed growth rate is applied and the model is set to fixed growth state
        - The model is checked for mass and charge balance, warning the user if unbalanced reactions exist

        :return: The community metabolic model
        """
        merged_model = None
        biomass_mets = {}
        idx = 0
        for model in self.member_models:
            idx += 1
            if idx == 1:
                merged_model = model.prepare_for_merging(shared_compartment_name=self.shared_compartment_name,
                                                         max_flux=self.max_flux)
                biomass_met_id = model.biomass_met.id
                biomass_met = merged_model.metabolites.get_by_id(biomass_met_id)
                biomass_mets[model.name] = biomass_met

                rxn = cobra.Reaction(f"{model.name}_to_community_biomass")
                rxn.add_metabolites({biomass_met: -1})
                merged_model.add_reactions([rxn])
                self.create_fraction_reaction(merged_model, member_name=model.name)

            else:
                extended_model = model.prepare_for_merging(shared_compartment_name=self.shared_compartment_name,
                                                           max_flux=self.max_flux)

                unbalanced_metabolites = check_mass_balance_of_metabolites_with_identical_id(extended_model,
                                                                                             merged_model)
                for met_id in unbalanced_metabolites:
                    met_base_name = get_metabolite_id_without_compartment(extended_model.metabolites.get_by_id(met_id))
                    logger.warning(
                        f"Matching of the metabolite {met_base_name} is unbalanced (mass and/or charge). "
                        f"Please manually curate this metabolite for a mass and charge balanced model!")
                no_annotation_overlap = check_annotation_overlap_of_metabolites_with_identical_id(extended_model,
                                                                                                  merged_model)
                self.no_annotation_overlap.extend(no_annotation_overlap)

                biomass_met_id = model.biomass_met.id

                biomass_met = extended_model.metabolites.get_by_id(biomass_met_id)
                biomass_mets[model.name] = biomass_met
                rxn = cobra.Reaction(f"{model.name}_to_community_biomass")
                rxn.add_metabolites({biomass_met: -1})
                extended_model.add_reactions([rxn])
                self.create_fraction_reaction(extended_model, member_name=model.name)

                regex_filter_cobra = RegexFilter(r"Ignoring")

                with temporary_logger_filter("cobra.core.model", regex_filter_cobra):
                    merged_model.merge(extended_model)
                biomass_mets[model.name] = merged_model.metabolites.get_by_id(biomass_met_id)

        if len(self.no_annotation_overlap) > 0:
            logger.warning(f"No annotation overlap found for matching several metabolites "
                           f"({len(self.no_annotation_overlap)}). Please make sure that the matched metabolites "
                           f"are indeed representing the same substance in all models! The list of metabolites"
                           f"without annotation overlap can be accessed via 'model.no_annotation_overlap'")

        self.fixed_growth_rate_flag = True
        self.fixed_abundance_flag = False
        self.merge_fraction_reactions(merged_model)
        self._add_fixed_abundance_reaction(merged_model)

        biomass_met = cobra.Metabolite(f"cpd11416_{self.shared_compartment_name}", name='Community Biomass',
                                       compartment=self.shared_compartment_name)
        biomass_rxn = cobra.Reaction("community_biomass")
        biomass_rxn.add_metabolites({biomass_met: -1})
        merged_model.add_reactions([biomass_rxn])
        # create additional reactions for each biomass reaction of a suborganism
        for member, met in biomass_mets.items():
            rxn = merged_model.reactions.get_by_id(f"{member}_to_community_biomass")
            replace_metabolite_stoichiometry(rxn, {met: -1, biomass_met: 1})
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

        self._model = merged_model

        if not self.is_mass_balanced():
            logger.warning(
                "Not all reactions in the model are mass and charge balanced. To check which reactions are "
                "imbalanced, please run the get_unbalanced_reactions method of this CommunityModel object")

        return merged_model

    def create_fraction_reaction(self, model, member_name):
        """
        Creates a reaction producing dummy metabolites, which control the bounds of the regular reactions. This method
        is used on the member metabolic models during community metabolic model generation.

        :param model: Model to be changed
        :param member_name: The name of the community member
        """
        fraction_reaction = cobra.Reaction(f"{member_name}_fraction_reaction")
        # create f_final metabolite
        f_final_met = cobra.Metabolite("f_final_met", name='Final Fraction Reaction Metabolite',
                                       compartment='fraction_reaction')
        fraction_reaction.add_metabolites({f_final_met: 1})
        # create biomass_metabolite for fraction reaction
        f_biomass_met = cobra.Metabolite(f'{member_name}_f_biomass_met', name=f'Fraction Biomass Metabolite of '
                                                                              f'{member_name}',
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
        """
        This function creates dummy metabolites for the constraints of each reaction in a model. This method
        is used on the member metabolic models during community metabolic model generation.

        :param model: Model to be changed
        :param member_name: The name of the community member
        :return: Dictionary of dummy metabolites (keys) and the bound they represent (values)
        """
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
                    fraction_reaction_mets[met_lb] = -coefficient * self._dummy_metabolite_scaling_factor
                    reaction.add_metabolites({met_lb: self._dummy_metabolite_scaling_factor})
                if reaction.upper_bound != 0:
                    coefficient = self.max_flux if reaction.upper_bound > self.max_flux else reaction.upper_bound
                    constrained_mets[met_ub] = coefficient
                    fraction_reaction_mets[met_ub] = coefficient * self._dummy_metabolite_scaling_factor
                    reaction.add_metabolites({met_ub: -self._dummy_metabolite_scaling_factor})

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

    def change_reaction_bounds(self, reaction: Union[str, cobra.Reaction], lower_bound, upper_bound):
        """
        This function allows changing reaction bounds in the community metabolic models. It takes care of adjusting the
        dummy metabolite stoichiometry.

        :param reaction: The reaction whose bounds should be set
        :param lower_bound: The new lower bound
        :param upper_bound: The new upper bound
        """
        model = self.model

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
            fraction_reaction_mets[met_lb] = -coefficient * self._dummy_metabolite_scaling_factor
            replace_metabolite_stoichiometry(reaction, {met_lb: self._dummy_metabolite_scaling_factor})
        else:
            replace_metabolite_stoichiometry(reaction, {met_lb: 0})
            fraction_reaction_mets[met_lb] = 0

        if upper_bound != 0:
            coefficient = self.max_flux if upper_bound > self.max_flux else upper_bound
            fraction_reaction_mets[met_ub] = coefficient * self._dummy_metabolite_scaling_factor
            replace_metabolite_stoichiometry(reaction, {met_ub: -self._dummy_metabolite_scaling_factor})
        else:
            fraction_reaction_mets[met_ub] = 0
            replace_metabolite_stoichiometry(reaction, {met_ub: 0})

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
        replace_metabolite_stoichiometry(fraction_reaction, fraction_reaction_mets)

        # Add sink reactions for fraction mets
        self.add_sink_reactions_to_metabolites(model, metabolites_needing_sink_reactions)

    def add_sink_reactions_to_metabolites(self, model, constraint_mets, lb=0., inplace=True):
        """
        Adds sink reactions for dummy metabolites.

        :param model: Model to be changed
        :param constraint_mets: Metabolites where sink reactions should be added
        :param lb: Lower bound of the sink reaction
        :param inplace: If true, the input model is changed, otherwise a copy is created
        :return: If the inplace flag is set to False, the updated model is returned
        """
        sink_max_flux = self.max_flux

        if sink_max_flux < 10 * self._dummy_metabolite_scaling_factor * self.max_flux:
            sink_max_flux = 10 * self._dummy_metabolite_scaling_factor * self.max_flux

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
        """
        Combines the fraction reactions of each of the community members

        :param merged_model: The community metabolic model to be changed
        """
        # create f_final reaction
        # This ensures that the fractions sum up to 1
        f_final = cobra.Reaction("f_final", name="final fraction reaction")
        f_final.bounds = (1, 1)
        f_final_met = merged_model.metabolites.get_by_id("f_final_met")
        f_final.add_metabolites({f_final_met: -1})
        merged_model.add_reactions([f_final])

    def apply_fixed_growth_rate(self, flux, model=None):
        """
        Applies a new fixed growth rate to the model. The model needs to be in fixed growth state, to allow this
        operation.

        :param flux: The new growth rate
        :param model: The model to update
        :return: The updated model
        """
        if not self.fixed_growth_rate_flag:
            logger.error("Error: The model needs to be in fixed growth rate structure to set a fixed growth rate.")
            return
        self.mu_c = flux

        if model is None:
            model = self.model
        model.reactions.get_by_id("community_biomass").bounds = (flux, flux)

        for member_name in self.get_member_names():
            fraction_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            try:
                fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                replace_metabolite_stoichiometry(fraction_rxn, {fraction_met: flux})
            except KeyError:
                fraction_met = self._backup_metabolites[f"{member_name}_f_biomass_met"]
                self.model.add_metabolites(fraction_met)
                replace_metabolite_stoichiometry(fraction_rxn, {fraction_met: flux})

        model.repair()

    def _add_fixed_abundance_reaction(self, model):
        """
        Adds an abundance reaction to the model for fixed abundance model structure.

        :param model: The model to be updated
        """
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
        """
        This function changes the model structure to fixed abundance, but variable growth rate. The model is left
        unchanged if it is already in fixed abundance structure.
        """
        if self.fixed_abundance_flag:
            logger.info(f"Note: Model already has fixed abundance structure.")
            return

        model = self.model

        # Remove the f_bio metabolites from the fraction reactions
        for member_name in self.get_member_names():
            fraction_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            try:
                fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                replace_metabolite_stoichiometry(fraction_rxn, {fraction_met: 0})
            except KeyError:
                fraction_met = self._backup_metabolites[f"{member_name}_f_biomass_met"]
                if fraction_met in fraction_rxn.metabolites:
                    self.model.add_metabolites(fraction_met)
                    replace_metabolite_stoichiometry(fraction_rxn, {fraction_met: 0})

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
        """
        Applying fixed abundance to the model. This is only available if the model is in fixed abundance structure
        (check fixed_abundance_flag).
        """
        if not self.fixed_abundance_flag:
            logger.error(
                "Error: the model is not in fixed abundance structure, but fixed abundance was tried to be applied. "
                "Convert the model to fixed abundance structure first.")
            return

        # Check if organism names are in the model
        if not all([name in self.get_member_names() for name in abd_dict.keys()]):
            err_msg = f"Error: Some names in the abundances are not part of the model." \
                      f"\n\tAbundances: {abd_dict.keys()}" \
                      f"\n\tOrganisms in model: {self.get_member_names()}"
            raise ValueError(err_msg)

        # Check that abundances sum to 1
        logger.debug(f"Checking if the sum of abundances is 1: {abd_dict}")
        if not np.isclose([sum(abd_dict.values())], [1.]):
            logger.warning(f"Warning: Abundances do not sum up to 1. Correction will be applied: {abd_dict}")
            if sum(abd_dict.values()) == 0.:
                logger.error(f"Error: The sum of abundances is 0")
                raise ValueError
            logger.debug("Applying abundance correction")
            correction_factor = 1 / sum(abd_dict.values())
            for name, abundance in abd_dict.items():
                new_abundance = abundance * correction_factor
                abd_dict[name] = new_abundance
            logger.info(f"Correction applied. New abundances are:\n{abd_dict}")
            if not np.isclose([sum(abd_dict.values())], [1.]):
                raise ValueError(f"Abundances do not sum up to 1: {abd_dict}")
            if np.isnan(list(abd_dict.values())).any():
                raise ValueError(f"Abundances contain NaN values: {abd_dict}")

        logger.debug("Extending abundances to include all organisms")
        # Extend abundances to include all organisms of model
        for name in self.get_member_names():
            if name not in abd_dict.keys():
                abd_dict[name] = 0.

        # Apply the abundance as ratios of f_biomass metabolites
        logger.debug(f"Applying abundances to model. Final abundances: {abd_dict}")
        model = self.model
        abd_rxn_mets = {}
        for member_name, fraction in abd_dict.items():
            if fraction < cobra.Configuration().tolerance and fraction != 0.:
                logger.warning(f"Abundance of {member_name} is lower than the solver tolerance:" 
                               f"{fraction} < {cobra.Configuration().tolerance}. "
                               f"This can lead to problems with the solver!")
            logger.debug(f"Setting abundance for {member_name}")
            try:
                f_bio_met = model.metabolites.get_by_id(f'{member_name}_f_biomass_met')
            except KeyError:
                f_bio_met = self._backup_metabolites[f"{member_name}_f_biomass_met"]
            f_rxn = model.reactions.get_by_id(f"{member_name}_fraction_reaction")
            f_rxn.bounds = (0., fraction)
            abd_rxn_mets[f_bio_met] = fraction

        logger.debug("Getting abundance reaction")
        abd_rxn = model.reactions.get_by_id("abundance_reaction")
        logger.debug("Setting abundance reaction")
        replace_metabolite_stoichiometry(abd_rxn, abd_rxn_mets)

        self._abundance_dict = abd_dict

        model.repair()

        return

    def convert_to_fixed_growth_rate(self, mu_c=None):
        """
        This function changes the model structure to fixed growth rate, but variable abundance profile. The model
        is left unchanged if it is already in fixed abundance structure.
        """
        if self.fixed_growth_rate_flag:
            logger.info(f"Note: Model already has fixed growth rate structure.")
            return

        model = self.model

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
        """
        Creates a dictionary with community member names as keys and equal fractions as values

        :return: Dictionary with community member names as keys and equal fractions as values
        """
        abundances = {}
        names = self.get_member_names()
        for name in names:
            abundances[name] = 1. / len(names)
        return abundances

    def equal_abundance(self):
        """
        Converts the model into fixed abundance state, if it is not already in this state and applies an abundance
        profile of equal abundance.
        """
        if not self.fixed_abundance_flag:
            self.convert_to_fixed_abundance()
        abundances = self.generate_equal_abundance_dict()
        self.apply_fixed_abundance(abundances)

    def convert_to_model_without_fraction_metabolites(self):
        """
        Converts the community metabolic model into an equal model, without dummy metabolites and reactions. This
        process essentially reverses the equivalent, bound free model structure applied to the models of the community
        members.

        :return: An equal model, without dummy metabolites and reactions
        """
        was_fixed_growth = False
        if self.fixed_growth_rate_flag:
            was_fixed_growth = True
            self.convert_to_fixed_abundance()

        model = self.model.copy()

        if was_fixed_growth:
            self.convert_to_fixed_growth_rate()

        reactions_to_remove = [model.reactions.get_by_id("f_final")]

        for reaction in model.reactions:
            if "fraction_reaction" in reaction.id:
                reactions_to_remove.append(reaction)
                for metabolite, coeff in reaction.metabolites.items():
                    if "_lb" == metabolite.id[-3:]:
                        rxn = model.reactions.get_by_id(metabolite.id[:-3])
                        rxn.lower_bound = -coeff / self._dummy_metabolite_scaling_factor
                        replace_metabolite_stoichiometry(rxn, {metabolite: 0})
                        replace_metabolite_stoichiometry(reaction, {metabolite: 0})
                        metabolite.remove_from_model(True)
                    elif "_ub" == metabolite.id[-3:]:
                        rxn = model.reactions.get_by_id(metabolite.id[:-3])
                        rxn.upper_bound = coeff / self._dummy_metabolite_scaling_factor
                        replace_metabolite_stoichiometry(rxn, {metabolite: 0})
                        replace_metabolite_stoichiometry(reaction, {metabolite: 0})
                        metabolite.remove_from_model(True)
                    elif "_f_biomass_met" in metabolite.id:
                        rxn = model.reactions.get_by_id(
                            metabolite.id.split("_f_biomass_met")[0] + "_to_community_biomass")
                        replace_metabolite_stoichiometry(rxn, {metabolite: 0})
                        replace_metabolite_stoichiometry(reaction, {metabolite: 0})
                        metabolite.remove_from_model(True)

        for reaction in reactions_to_remove:
            reaction.remove_from_model(remove_orphans=True)

        return model

    def load_medium_from_file(self, file_path):
        """
        Loads a medium for the community metabolic model from file and sets it as the medium attribute of this object.
        The file needs to be in the following format: A csv file with two columns separated by a comma (,). The two
        columns are named compounds and maxFlux. The compounds columns contains the metabolite IDs of the boundary
        metabolites as in the community metabolic model. The maxFlux column contains the maximum influx of the
        respective value as an integer or float.

        :param file_path: The path to the medium file
        """
        # load the medium dictionary
        medium_dict = read_medium_from_file(file_path, comp=f"_{self.shared_compartment_name}")
        self.medium = medium_dict

    def apply_medium(self, medium=None):
        """
        Applies the medium that is specified in the medium attribute to the community metabolic model

        :param medium: Optional. A dictionary with exchange reaction IDs as keys and the maximum
            flux of the respective metabolite as value. If none is given, the medium in self.medium is applied.
        :return: The updated model
        """

        if medium is not None:
            self.medium = medium

        test_if_medium_exists = self.medium
        medium_model = self.model
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

        self._model = medium_model
        self.medium_flag = True
        self.model.reactions.abundance_reaction.bounds = 0.,1000.
        return medium_model

    def run_fba(self):
        """
        Optimizes the community metabolic model with flux balance analysis. Returns the flux vector of the solution as
        a dataframe.

        :return: Dataframe of the solution flux vector
        """
        solution = self.model.optimize()
        solution_df = solution.fluxes.to_frame()
        solution_df.insert(loc=0, column='reaction', value=list(solution.fluxes.index))
        solution_df.columns = ["reaction_id", "flux"]

        return solution_df

    def fba_solution_flux_vector(self, file_path=""):
        """
        Runs flux balance analysis on the community metabolic model and saves the solution flux vector as a csv file.

        :param file_path: Path of the output file
        :return: Dataframe of the solution flux vector
        """
        solution_df = self.run_fba()

        if len(file_path) > 0:
            logger.info(f"Saving flux vector to {file_path}")
            solution_df.to_csv(file_path, sep="\t", header=True, index=False, float_format='%f')
        return solution_df

    def loopless_fva(self,
                     reaction_ids,
                     fraction_of_optimum=None,
                     use_loop_reactions_for_ko=True,
                     ko_candidate_ids=None,
                     verbose=False,
                     processes=None):
        return loopless_fva(self,
                            reaction_ids,
                            fraction_of_optimum=fraction_of_optimum,
                            use_loop_reactions_for_ko=use_loop_reactions_for_ko,
                            ko_candidate_ids=ko_candidate_ids,
                            verbose=verbose,
                            processes=processes)

    def run_fva(self, fraction_of_optimum=0.9, composition_agnostic=False, loopless=False, fva_mu_c=None,
                only_exchange_reactions=True, reactions=None, verbose=False, processes=None):
        """
        Run flux variability on the community metabolic model. By default, only reactions connected to metabolites in
        the shared exchange compartment are analysed.

        :param fraction_of_optimum: The fraction of the optimal objective flux that needs to be reached
        :param composition_agnostic: Removes constrains set by fixed growth rate or fixed abundance. This also allows
            solutions without balanced growth, i.e. different growth rate of community members.
        :param loopless: Avoids loops in the solutions, but takes longer to compute
        :param fva_mu_c: Set a temporary community growth rate for the community metabolic model
        :param only_exchange_reactions: Analyse only reactions connected to metabolites in the shared exchange
            compartment
        :param reactions: A list of reactions that should be analysed. This parameter is overwritten if
            only_exchange_reactions is set to True
        :param verbose: Print progress of loopless FVA
        :param processes: The number of processes to use
        :return: A dataframe of reaction flux solution ranges. Contains the columns reaction_id, min_flux and max_flux
        """
        logger.debug(f"Starting run FVA with parameters:\n\t"
                     f"fraction_of_optimum: {fraction_of_optimum}\n\t"
                     f"composition_agnostic: {composition_agnostic}\n\t"
                     f"loopless: {loopless}\n\t"
                     f"fva_mu_c: {fva_mu_c}\n\t"
                     f"only_exchange_reactions: {only_exchange_reactions}\n\t"
                     f"reactions: {reactions}\n\t"
                     f"verbose: {verbose}\n\t"
                     f"processes: {processes}\n\t")
        model = self.model

        if fva_mu_c is None and composition_agnostic:
            logger.debug(f"Setting fva_mu_c to 0 and fraction_of_optimum to 0")
            fva_mu_c = 0.
            fraction_of_optimum = 0.
        elif fva_mu_c is not None:
            logger.debug("Setting fraction_of_optimum to 1")
            fraction_of_optimum = 1.

        if only_exchange_reactions:
            if verbose:
                logger.info(f"Setting reactions to be analysed to exchange reactions only")
            else:
                logger.debug("Setting reactions to be analysed to exchange reactions only")
            reactions = model.reactions.query(lambda x: any([met.compartment == self.shared_compartment_name
                                                             for met in x.metabolites.keys()]))
        elif reactions is None:
            if verbose:
                logger.info(f"Setting reactions to be analysed to all non-fraction-reactions")
            else:
                logger.debug("Setting reactions to be analysed to all non-fraction-reactions")
            reactions = model.reactions.query(lambda x: x not in self.f_reactions)

        if fva_mu_c is not None:
            logger.debug(f"fva_mu_c is not None: {fva_mu_c}")
            f_bio_mets = {}
            if self.fixed_growth_rate_flag:
                logger.debug(f"Model is in fixed growth rate")
                mu_c = self.mu_c
                self.apply_fixed_growth_rate(fva_mu_c)
                if composition_agnostic:
                    logger.debug("Prepare model for composition agnostic run")
                    # Allow flux through the biomass reaction
                    self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=self.max_flux)
                    # Uncouple the biomass flux from the fraction reactions. This allows a model structure where member
                    # organisms have their fluxes scaled by abundance, but their growth rate is not equal. This allows
                    # to discover a superset of possible metabolite exchanges.
                    for member_name in self.get_member_names():
                        biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                        fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                        f_bio_mets[member_name] = fraction_met
                        replace_metabolite_stoichiometry(biomass_rxn, {fraction_met: 0})

                if loopless:
                    if verbose:
                        logger.info(f"Running loopless FVA")
                    else:
                        logger.debug("Running loopless FVA")
                    solution_df = self.loopless_fva(reactions,
                                                    fraction_of_optimum=fraction_of_optimum,
                                                    use_loop_reactions_for_ko=True,
                                                    verbose=verbose,
                                                    processes=processes)
                else:
                    if verbose:
                        logger.info(f"Running FVA")
                    else:
                        logger.debug("Running FVA")
                    solution_df = fva(self,
                                      reactions,
                                      fraction_of_optimum=fraction_of_optimum,
                                      processes=processes)

                logger.debug("FVA finished!")
                # Revert changes
                if composition_agnostic:
                    self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=0.)
                    for member_name, fraction_met in f_bio_mets.items():
                        biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                        biomass_rxn.add_metabolites({fraction_met: -1}, combine=True)

                self.apply_fixed_growth_rate(mu_c)
            else:
                self.convert_to_fixed_growth_rate(mu_c=fva_mu_c)

                if composition_agnostic:
                    # Allow flux through the biomass reaction
                    self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=self.max_flux)

                    # Uncouple the biomass flux from the fraction reactions. This allows a model structure where member
                    # organisms have their fluxes scaled by abundance, but their growth rate is not equal. This allows
                    # to discover a superset of possible metabolite exchanges.
                    for member_name in self.get_member_names():
                        biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                        fraction_met = model.metabolites.get_by_id(f"{member_name}_f_biomass_met")
                        f_bio_mets[member_name] = fraction_met
                        replace_metabolite_stoichiometry(biomass_rxn, {fraction_met: 0})

                if loopless:
                    if verbose:
                        logger.info(f"Running loopless FVA")
                    solution_df = self.loopless_fva(reactions,
                                                    fraction_of_optimum=fraction_of_optimum,
                                                    use_loop_reactions_for_ko=True,
                                                    verbose=verbose,
                                                    processes=processes)
                else:
                    if verbose:
                        logger.info(f"Running FVA")
                    solution_df = fva(self,
                                      reactions,
                                      fraction_of_optimum=fraction_of_optimum,
                                      processes=processes)
                # Revert changes
                if composition_agnostic:
                    self.change_reaction_bounds("community_biomass", lower_bound=0., upper_bound=0.)
                    for member_name, fraction_met in f_bio_mets.items():
                        biomass_rxn = model.reactions.get_by_id(f"{member_name}_to_community_biomass")
                        biomass_rxn.add_metabolites({fraction_met: -1}, combine=True)

                self.convert_to_fixed_abundance()
        else:
            logger.debug(f"fva_mu_c is None: {fva_mu_c}")
            if loopless:
                if verbose:
                    logger.info(f"Running loopless FVA")
                else:
                    logger.debug(f"Running loopless FVA")
                solution_df = self.loopless_fva(reactions,
                                                fraction_of_optimum=fraction_of_optimum,
                                                use_loop_reactions_for_ko=True,
                                                verbose=verbose,
                                                processes=processes)
            else:
                if verbose:
                    logger.info(f"Running FVA")
                else:
                    logger.debug(f"Running FVA")
                solution_df = fva(self,
                                  reactions,
                                  fraction_of_optimum=fraction_of_optimum,
                                  processes=processes)
            logger.debug("FVA finished")

        solution_df.insert(loc=0, column='reaction', value=list(solution_df.index))
        solution_df.columns = ["reaction_id", "min_flux", "max_flux"]
        return solution_df

    def fva_solution_flux_vector(self, file_path="",
                                 fraction_of_optimum=0.9,
                                 composition_agnostic=False,
                                 loopless=True,
                                 processes=None):
        """
        Run flux variability analysis on the current configuration of the community metabolic model and save the
        resulting flux ranges to a csv file.

        :param file_path: The fraction of the optimal objective flux that needs to be reached
        :param fraction_of_optimum: Path of the output file
        :param composition_agnostic: Run FVA with relaxed constraints (composition agnostic)
        :param loopless: Run loopless FVA
        :param processes: The number of processes to use
        :return: A dataframe of reaction flux solution ranges. Contains the columns reaction_id, min_flux and max_flux
        """
        solution_df = self.run_fva(fraction_of_optimum=fraction_of_optimum,
                                   processes=processes,
                                   composition_agnostic=composition_agnostic,
                                   loopless=loopless)

        if len(file_path) > 0:
            logger.info(f"Saving flux vector to {file_path}")
            solution_df.to_csv(file_path, sep="\t", header=True, index=False, float_format='%f')
        return solution_df

    def cross_feeding_metabolites_from_fba(self, return_flux_vector=False):
        """
        Run flux balance analysis and convert the solution flux vector into a table of metabolites, including the
        solution flux for the exchange reaction of each metabolite for every community member.

        :param return_flux_vector: If True, return the flux vector from FBA output as well
        :return: A dataframe of the solution flux for the exchange reaction of each metabolite for every community
            member
        """
        model = self.model

        solution_df = self.run_fba()
        rows = []

        exchg_metabolites = model.metabolites.query(lambda x: x.compartment == self.shared_compartment_name)
        member_names = self.get_member_names()

        transfer_reactions = self.transfer_reactions
        for exchg_met in exchg_metabolites:
            # Check flux of transfer ("_TF_") reactions to organism
            row_dict = {"metabolite_id": exchg_met.id, "metabolite_name": exchg_met.name}
            for name in member_names:
                row_dict[name] = 0.
            for rxn in exchg_met.reactions:
                if rxn not in transfer_reactions:
                    continue
                rxn_member = ""
                for member in self.get_member_names():
                    if len(member) <= len(rxn.id) and member in rxn.id[:len(member)]:
                        rxn_member = member
                        break
                if rxn_member not in member_names:
                    raise ValueError(f"Community member extracted from reaction is not part of the model")
                if rxn.id not in set(solution_df["reaction_id"]):
                    raise ValueError(f"Reaction is not part of the solution dataframe")
                flux = float(solution_df.loc[rxn.id, "flux"])
                row_dict[rxn_member] = 0. if close_to_zero(flux) else flux
            rows.append(row_dict)

        exchg_metabolite_df = pd.DataFrame(rows,
                                           columns=["metabolite_id", "metabolite_name"].extend(list(member_names))
                                           )
        cross_feeding_metabolites = exchg_metabolite_df.copy()
        cross_feeding_metabolites.drop(columns=["metabolite_id", "metabolite_name"], inplace=True)
        cross_feeding_mask = cross_feeding_metabolites.apply(lambda x: any(x > 0.) and any(x < 0.), axis=1)
        exchg_metabolite_df["cross_feeding"] = cross_feeding_mask
        if return_flux_vector:
            return exchg_metabolite_df, solution_df
        return exchg_metabolite_df

    def cross_feeding_metabolites_from_fva(self, fraction_of_optimum=0.,
                                           composition_agnostic=False,
                                           fva_mu_c=None,
                                           loopless=True,
                                           processes=None,
                                           return_flux_vector=False):
        """
        Run flux variability analysis and convert the solution flux ranges into a table of metabolites, including the
        solution flux ranges for the exchange reaction of each metabolite for every community member.

        :param fraction_of_optimum: The fraction of the optimal objective flux that needs to be reached
        :param composition_agnostic: Removes constrains set by fixed growth rate or fixed abundance. This also allows
            solutions without balanced growth, i.e. different growth rate of community members.
        :param fva_mu_c: Set a temporary community growth rate for the community metabolic model
        :param loopless: Run loopless FVA
        :param processes: The number of processes to use
        :param return_flux_vector: If True, return the flux vector from FVA output as well
        :return: A dataframe of the solution flux range for the exchange reaction of each metabolite for every community
            member
        """
        model = self.model

        solution_df = self.run_fva(fraction_of_optimum=fraction_of_optimum,
                                   composition_agnostic=composition_agnostic, fva_mu_c=fva_mu_c,
                                   only_exchange_reactions=True,
                                   loopless=loopless,
                                   processes=processes)
        rows = []
        exchg_metabolites = model.metabolites.query(lambda x: x.compartment == self.shared_compartment_name)
        member_names = self.get_member_names()

        columns = ["metabolite_id", "metabolite_name", "cross_feeding"]
        for name in member_names:
            columns.append(name + "_min_flux")
            columns.append(name + "_max_flux")

        transfer_reactions = self.transfer_reactions
        for exchg_met in exchg_metabolites:
            # Check flux of transfer ("_TF_") reactions to organism
            row_dict = {"metabolite_id": exchg_met.id, "metabolite_name": exchg_met.name, "cross_feeding": False}
            for name in member_names:
                row_dict[name + "_min_flux"] = 0.
                row_dict[name + "_max_flux"] = 0.
            for rxn in exchg_met.reactions:
                if rxn not in transfer_reactions:
                    continue
                rxn_member = ""
                for member in self.get_member_names():
                    if len(member) <= len(rxn.id) and member in rxn.id[:len(member)]:
                        rxn_member = member
                        break
                if rxn_member not in member_names:
                    raise ValueError(f"Community member extracted from reaction is not part of the model")
                if rxn.id not in set(solution_df["reaction_id"]):
                    raise ValueError(f"Reaction is not part of the solution dataframe")
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

        exchg_metabolite_df = pd.DataFrame(rows, columns=columns)
        if return_flux_vector:
            return exchg_metabolite_df, solution_df
        return exchg_metabolite_df

    def format_exchg_rxns(self, exchg_metabolite_df):
        """
        Formats the solution flux dataframe of FBA or FVA to a dataframe that contains which community member consumes
        or produces a given metabolite in the solution

        :param exchg_metabolite_df: The solution flux dataframe
        :return: The formatted dataframe
        """
        rows = []
        member_names = self.get_member_names()

        for idx, row in exchg_metabolite_df.iterrows():
            row_dict = {"metabolite_id": row["metabolite_id"], "metabolite_name": row["metabolite_name"],
                        "cross_feeding": row["cross_feeding"], "produced_by": [], "consumed_by": []}
            for member in member_names:
                if member in row.keys():
                    if row[member] < 0.:
                        row_dict["consumed_by"].append(member)
                    if row[member] > 0.:
                        row_dict["produced_by"].append(member)
                else:
                    if row[member + "_min_flux"] < 0.:
                        row_dict["consumed_by"].append(member)
                    if row[member + "_max_flux"] > 0.:
                        row_dict["produced_by"].append(member)
            rows.append(row_dict)

        columns = ["metabolite_id", "metabolite_name", "cross_feeding", "produced_by", "consumed_by"]

        exchg_metabolite_df = pd.DataFrame(rows, columns=columns)

        return exchg_metabolite_df

    def potential_metabolite_exchanges(self,
                                       fba=False,
                                       composition_agnostic=True,
                                       fva_mu_c=None,
                                       fraction_of_optimum=0.,
                                       loopless=True,
                                       processes=None,
                                       return_flux_vector=False):
        """
        Calculates all potentially exchanged metabolites between the community members. This can be done via flux
        balance analysis or flux variability analysis.

        :param fba: If true, flux balance analysis, otherwise flux variability analysis is used
        :param composition_agnostic: Removes constrains set by fixed growth rate or fixed abundance. This also allows
            solutions without balanced growth, i.e. different growth rate of community members.
        :param fva_mu_c: Set a temporary community growth rate for the analysis (only FVA).
        :param fraction_of_optimum: The fraction of the optimal objective flux that needs to be reached (only FVA)
        :param loopless: Run loopless FVA
        :param processes: The number of processes to use (only FVA)
        :param return_flux_vector: If True, return the flux vector from FBA/FVA output as well
        :return: A dataframe of which metabolites are cross-fed, taken up or secreted by each community member
        """

        if fba:
            exchange_df, flux_vector = self.cross_feeding_metabolites_from_fba(return_flux_vector=True)
        elif composition_agnostic:
            exchange_df, flux_vector = self.cross_feeding_metabolites_from_fva(fraction_of_optimum=1.,
                                                                               fva_mu_c=None,
                                                                               composition_agnostic=True,
                                                                               loopless=loopless,
                                                                               processes=processes,
                                                                               return_flux_vector=True)
        else:
            exchange_df, flux_vector = self.cross_feeding_metabolites_from_fva(fraction_of_optimum=fraction_of_optimum,
                                                                               fva_mu_c=fva_mu_c,
                                                                               composition_agnostic=False,
                                                                               loopless=loopless,
                                                                               processes=processes,
                                                                               return_flux_vector=True)

        if return_flux_vector:
            return self.format_exchg_rxns(exchange_df), flux_vector
        return self.format_exchg_rxns(exchange_df)

    def report(self, verbose=True, max_reactions=5000):
        """
        This function gives a report on the community metabolic model. It includes information on the number of
        metabolites, reactions and genes, the names and number of community members, the model objective and the
        presence of mass or charge unbalanced reactions and thermodynamically infeasible cycles.

        :param verbose: Prints the report
        :param max_reactions: Excludes calculation of thermodynamically infeasible cycles from the report if the number
            of reactions in the model exceeds the number specified in this parameter. If the parameter is set to None,
            thermodynamically infeasible cycles are calculated regardless of model size
        :return: A dictionary of model statistics
        """
        report_dict = {}
        model_structure = "fixed growth rate" if self.fixed_growth_rate_flag else "fixed abundance"
        num_metabolites = len(self.model.metabolites)
        num_f_metabolites = len(self.f_metabolites)
        num_model_metabolites = num_metabolites - num_f_metabolites
        num_reactions = len(self.model.reactions)
        num_f_reactions = len(self.f_reactions)
        num_model_reactions = num_reactions - num_f_reactions
        num_genes = len(self.model.genes)
        member_names = self.get_member_names()
        num_members = len(member_names)
        objective_expression = self.model.objective.expression
        objective_direction = self.model.objective.direction
        unbalanced_reactions = self.get_unbalanced_reactions()
        num_unbalanced_reactions = len(unbalanced_reactions)

        reactions_in_loops = "NaN"
        num_loop_reactions = "NaN"
        if max_reactions is not None and num_model_reactions <= max_reactions:
            reactions_in_loops = self.get_loops()
            num_loop_reactions = len(reactions_in_loops)
        else:
            logger.info(
                f"Note: The model has more than {max_reactions} reactions. Calculation of loops is skipped, as this "
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
        Save the community model object as a SBML file. This also includes the names of the community members and their
        abundance (if set).

        :param file_path: The path to the output file
        """
        # Generate a libsbml.model object
        cobra.io.write_sbml_model(self.model, filename=file_path)
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
        """
        Loads a community metabolic model from SBML file generated by PyCoMo as CommunityModel object.

        :param file_path: The path to the SBML model file
        :return: The community metabolic model as CommunityModel object
        """
        abundance_parameters = get_abundance_parameters_from_sbml_file(file_path)
        if not (len(abundance_parameters) > 0):
            raise ValueError(f"PyCoMo community model parameters could not be extracted from SBML file.\n"
                             f"Make sure the SBML file is in a valid format for PyCoMo community metabolic models.")

        flags_and_muc = get_flags_and_muc_from_sbml_file(file_path)
        if not (len(flags_and_muc) == 4):
            raise ValueError(f"PyCoMo community model parameters could not be extracted from SBML file.\n"
                             f"Make sure the SBML file is in a valid format for PyCoMo community metabolic models.")

        constructor_args = {}
        constructor_args["member_names"] = list(abundance_parameters.keys())
        if any([val is not None for val in abundance_parameters.values()]):
            constructor_args["abundance_profile"] = abundance_parameters
        constructor_args["model"] = cobra.io.read_sbml_model(file_path)

        constructor_args["fixed_abundance"] = flags_and_muc["fixed_abundance_flag"]
        constructor_args["fixed_growth_rate"] = flags_and_muc["fixed_growth_rate_flag"]

        name = constructor_args["model"].id
        return cls(name=name, mu_c=flags_and_muc["mu_c"], **constructor_args)

    def feasible_composition_range(self, growth_rate=0.):
        """
        Computes the range of feasible compositions at a given growth rate. Changes the model into fixed growth-rate structure.

        :param growth_rate: the target growth rate
        :return: maximum growth-rate
        """
        names = self.get_member_names()
        frxns = [self.model.reactions.get_by_id(f'{name}_fraction_reaction') for name in names]

        # Set growth rate
        if growth_rate < 0. or np.isnan(growth_rate):
            raise ValueError(f"The growth rate needs to be equal to or larger than 0 (provided value: {growth_rate})")

        if self.fixed_abundance_flag:
            logger.debug("Converting to fixed growth-rate")
            self.convert_to_fixed_growth_rate()
        logger.debug(f"Setting fixed growth-rate to {growth_rate}")
        self.apply_fixed_growth_rate(growth_rate)

        # run fva
        fva_result = self.run_fva(only_exchange_reactions=False, reactions=frxns, fraction_of_optimum=1)
        # create new row containing mu_max
        new_row = pd.DataFrame({"reaction_id": ["community_biomass"], "min_flux": [growth_rate], "max_flux": [growth_rate]})
        # join fva_result with new row
        result = pd.concat([fva_result, new_row], ignore_index=True)
        result["min_flux"] = result["min_flux"].apply(lambda r: 0. if close_to_zero(r) else r)
        result["max_flux"] = result["max_flux"].apply(lambda r: 0. if close_to_zero(r) else r)
        logger.info(f"Feasible composition ranges are: {result}")

        return result


    def max_growth_rate(self, minimal_abundance=0, return_abundances=False, sensitivity=6):
        """
        Computes the overall maximum growth-rate of the community. Changes the model into fixed growth-rate structure,
        set to the overall maximum growth-rate.

        :param minimal_abundance: float indicating the minimal abundance of each member in the community
        :param return_abundances: If set to True, returns a dataframe with the ranges of feasible member abundances at
        the maximum growth rate
        :param sensitivity: How many decimal places should be calculated
        :return: maximum growth-rate
        """

        # set minimal abundance of members
        names = self.get_member_names()
        frxns = [self.model.reactions.get_by_id(f'{name}_fraction_reaction') for name in names]
        # check that the sum of minimal abundances is not greater than 1
        if len(names) * minimal_abundance > 1.0:
            raise ValueError("sum of abundances is greater than 1")
        for frxn in frxns:
            frxn.bounds = (minimal_abundance, 1.0)

        # set starting values
        lb = 0.
        ub = self.max_flux
        x = lb + 2*(10. ** (-sensitivity))
        result_difference = self.max_flux
        result = 0.
        x_is_fba_result = False

        while result_difference > 10. ** (-sensitivity):
            logger.info(f"New round: lb: {lb}, ub: {ub}, x: {x}")
            # calculate and set mu
            if self.fixed_abundance_flag:
                logger.debug("Converting to fixed growth-rate")
                self.convert_to_fixed_growth_rate()
            logger.debug(f"Setting fixed growth-rate to {x}")
            self.apply_fixed_growth_rate(x)

            # sometimes, through computational mistakes, a lower bound ends up smaller than the upper bound
            # in that case, the lower bound is the maximum growth rate
            logger.debug(f"Checking results if {lb} > {ub}")
            if lb > ub:
                logger.debug(f"Result is lb {lb}")
                result = lb
                break

            # check if mu is feasible
            logger.debug("Check if mu is feasible")
            try:
                # run fba
                logger.debug("Running FBA to see if infeasible")
                objective_value = self.model.slim_optimize()
                logger.debug(f"Objecive value after FBA: {objective_value}")
                if np.isnan(objective_value):
                    # FBA was infeasible
                    raise cobra.exceptions.Infeasible("Error: Infeasible!")
                # run fva
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    logger.debug("Running FVA")
                    df = self.run_fva(only_exchange_reactions=False, reactions=frxns, fraction_of_optimum=1)
                    logger.debug(f"FVA completed:\n{df}")
                # fix abundances to a point within the feasible composition space of the current mu
                # get name of the member for each row in the fva df
                member_id = [string.replace("_fraction_reaction", "") for string in df["reaction_id"]]
                # dataframes with min and max flux values of fraction reactions
                # Ensure lower bounds close to zero are rounded to zero.
                # This also helps with very small negative values.
                lb_df = (df["min_flux"]).apply(
                    lambda lb_x: 0. if close_to_zero(lb_x) else lb_x
                ).apply(
                    lambda lb_x: 1. if close_to_zero(1. - lb_x) else lb_x
                ).apply(
                    lambda lb_x: lb_x if lb_x > minimal_abundance else minimal_abundance
                )
                ub_df = (df["max_flux"]).apply(
                    lambda ub_x: 0. if close_to_zero(ub_x) else ub_x
                ).apply(
                    lambda ub_x: 1. if close_to_zero(1. - ub_x) else ub_x
                )
                logger.debug(df)
                logger.debug(f"lb_df: {lb_df}")
                logger.debug(f"ub_df: {ub_df}")
                # dataframe with range of possible abundances
                r_df = (ub_df - lb_df).apply(lambda r_x: 0. if close_to_zero(r_x) else r_x)
                logger.debug(f"R_df: {r_df}")
                # check that all upper bounds are bigger than the set minimal abundance
                if not (ub_df >= minimal_abundance).all():
                    logger.warning(f"Not all upper bounds are bigger than the set minimal abundance:\n{ub_df}")
                    raise ValueError("Not all upper bounds are bigger than the set minimal abundance")
                # check that upper bounds are larger than lower bounds and therefore that no value in R_df is negative
                if not (r_df >= 0.).all():
                    logger.warning(f"Some possible ranges are less than 0.:\n{r_df}")
                    raise ValueError("Some possible ranges are less than 0.")
                # check that the solution is not erroneous in the sense that the abundances sum up to 1.
                sum_ub_df = sum(ub_df)
                sum_ub_df = 1. if close_to_zero(1. - sum_ub_df) else sum_ub_df
                sum_lb_df = sum(lb_df)
                sum_lb_df = 0. if close_to_zero(sum_lb_df) else sum_lb_df
                sum_lb_df = 1. if close_to_zero(1. - sum_lb_df) else sum_lb_df
                if not (sum_ub_df >= 1. >= sum_lb_df):
                    logger.warning(f"Upper and lower bounds do not allow abundances to sum up to 1: {ub_df}, {lb_df}")
                    raise ValueError(f"Upper and lower bounds do not allow abundances to sum up to 1: {ub_df}, {lb_df}")
                # The total flexibility for different abundances
                delta = 1. - sum_lb_df
                delta = 0. if close_to_zero(delta) else delta
                logger.debug(f"Delta: {delta}")
                # if Delta is 0, the abundances must be the corresponding minimal fluxes in the fva
                if delta > 0.:
                    logger.debug(f"Sum R_df: {sum(r_df)}")
                    delta_df = delta * r_df / sum(r_df)
                    ab_df = lb_df + delta_df
                else:
                    ab_df = lb_df
                logger.debug(f"ab_df: {ab_df}")
                # create an abundance dictionary
                # Safeguard against abundances very close to 0. or 1
                abd_t = max(cobra.Configuration().tolerance * 100, 10 ** (-sensitivity))
                ab_df = ab_df.apply(lambda abd: 0. if close_to_zero(abd, t=abd_t) else (
                    1. if close_to_zero(1. - abd, t=abd_t) else abd))
                logger.debug(f"Corrected abundances: {ab_df}")
                abundance_dict = dict(zip(member_id, ab_df))
                # set abundance
                logger.debug(f"Setting abundance to {abundance_dict}")
                self.convert_to_fixed_abundance()
                self.apply_fixed_abundance(abundance_dict)

                # conduct fba with this composition
                logger.debug(f"Running FBA")
                fba_result = self.model.slim_optimize()
                if isnan(fba_result):
                    logger.debug("FBA result is NaN")
                    raise cobra.exceptions.Infeasible("FBA result is NaN!")
                logger.debug(f"fba results: {fba_result}")

                # check if fba result >= x
                if fba_result >= x:
                    # fba result becomes the new x
                    x = fba_result + 2 * 10 ** -sensitivity
                    x_is_fba_result = True
                    # adjust remaining parameters
                    lb = fba_result
                    result_difference = fba_result - result
                    result = fba_result
                elif close_to_zero(fba_result - x):
                    logger.debug(f"fba result is only marginally smaller than x: {fba_result} ~= {x}")
                    # fba result becomes the new x
                    x = fba_result + 2 * 10 ** -sensitivity
                    x_is_fba_result = True
                    # adjust remaining parameters
                    lb = fba_result
                    result_difference = fba_result - result
                    result = fba_result
                else:
                    # adjust upper bound
                    ub = fba_result

            except (ValueError, cobra.exceptions.Infeasible):
                # adjust parameters in case of infeasible fva solution
                logger.debug(f"Infeasible!")
                ub = x

            # update x
            if not x_is_fba_result:
                x = (ub + lb) / 2
            x_is_fba_result = False

            if ub - lb < 10 ** (-sensitivity):
                result_difference = 0.
            logger.debug(f"Loop end with result_difference {result_difference}")

        # truncate result
        result = np.floor((result * 10 ** sensitivity)) / 10 ** sensitivity
        if self.fixed_abundance_flag:
            self.convert_to_fixed_growth_rate()
        # set result as growth rate
        self.apply_fixed_growth_rate(result)

        if return_abundances:
            # run fva
            fva_result = self.run_fva(only_exchange_reactions=False, reactions=frxns, fraction_of_optimum=1)
            # create new row containing mu_max
            new_row = pd.DataFrame({"reaction_id": ["community_biomass"], "min_flux": [result], "max_flux": [result]})
            # join fva_result with new row
            result = pd.concat([fva_result, new_row], ignore_index=True)
            result["min_flux"] = result["min_flux"].apply(lambda r: 0. if close_to_zero(r) else r)
            result["max_flux"] = result["max_flux"].apply(lambda r: 0. if close_to_zero(r) else r)

        logger.info(f"Maximum growth-rate is {result}")
        return result


def doall(model_folder="",
          models=None,
          com_model=None,
          out_dir="",
          community_name="community_model",
          fixed_growth_rate=None,
          abundance="equal",
          medium=None,
          fba_solution_file=None,
          fva_solution_file=None,
          fva_solution_threshold=0.9,
          fba_interaction_file=None,
          fva_interaction_file=None,
          composition_agnostic=False,
          max_growth_rate_file=None,
          sbml_output_file=None,
          return_as_cobra_model=False,
          merge_via_annotation=None,
          loopless=True,
          num_cores=1):
    """
    This method is meant as an interface for command line access to the functionalities of PyCoMo. It includes
    generation of community metabolic models, their analyses and can save the results of analyses as well as the model
    to files.

    :param model_folder: Path to a directory containing metabolic models to be merged into a community
    :param models: A list of file paths to metabolic model files or a list of COBRApy model objects, to be merged into
        a community
    :param com_model: Path to a SBML file of a community metabolic model generated by PyCoMo
    :param out_dir: Path to an output directory
    :param community_name: The name of the generated community
    :param fixed_growth_rate: Sets the community metabolic model to fixed growth state with this value as growth rate
    :param abundance: Sets the community metabolic model to fixed abundance state. This parameter can be either None,
        equal, or an abundance dict (community member names as keys and fractions as values).
    :param medium: Path to a medium file to be applied to the community metabolic model
    :param fba_solution_file: Run FBA and save the solution to this file
    :param fva_solution_file: Run FVA and save the solution to this file
    :param fva_solution_threshold: The fraction of the objective optimum needed to be reached in FVA
    :param fba_interaction_file: Run FBA to calculate cross-feeding interactions and save the solution to this file
    :param fva_interaction_file: Run FVA to calculate cross-feeding interactions and save the solution to this file
    :param composition_agnostic: Run FVA with relaxed constraints (composition agnostic)
    :param sbml_output_file: If a filename is given, save the community metabolic model as SBML file
    :param max_growth_rate_file: If a filename is given, calculate the maximum community growth-rate, including the
        feasible community composition.
    :param return_as_cobra_model: If true, returns the community metabolic model as COBRApy model object, otherwise as
        PyCoMo CommunityModel object
    :param merge_via_annotation: The database to be used for matching boundary metabolites when merging into a
        community metabolic model. If None, matching of metabolites is done via metabolite IDs instead
    :param loopless: Run loopless FVA
    :param num_cores: The number of cores to use in flux variability analysis
    :return: The community metabolic model, either as COBRApy model object or PyCoMo CommunityModel object (see
        return_as_cobra_model parameter)
    """
    com_model_obj = None
    if com_model is not None:
        # Load community model
        logger.info(f"Loading community model from file {com_model}")
        com_model_obj = CommunityModel.load(com_model)
    else:
        # Load single organism models
        # Either from folder or as list of file names or list of cobra models
        if model_folder != "":
            logger.info(f"Loading member models from folder {model_folder}")
            named_models = load_named_models_from_dir(model_folder)
            if len(named_models) == 0:
                raise ValueError(
                    f"No models found in the input models' folder. \nPlease make sure the selected folder contains "
                    f"metabolic model files.")
        elif not isinstance(models, list) or len(models) == 0:
            raise ValueError(f"No models supplied to the doall function. Please supply either a path to the folder "
                             f"containing the models to the model_folder variable or a list of filepaths or cobra "
                             f"models"
                             f"to the models variable.")
        elif all(list(map(lambda x: isinstance(x, cobra.Model), models))):
            # Extract names and store in named models
            named_models = {model.name: model for model in models}
        elif all(list(map(lambda x: isinstance(x, str), models))):
            logger.info(f"Loading member models from files")
            named_models = {}
            for model_path in models:
                model, name = load_named_model(model_path)
                named_models[name] = model
        else:
            raise TypeError(f"Input models are either of mixed type or neither filepath, nor cobra model.")

        # Create single organism models
        single_org_models = [SingleOrganismModel(model, name) for name, model in named_models.items()]

        # Create a community model
        logger.info(f"Creating a community model")
        com_model_obj = CommunityModel(single_org_models, community_name, merge_via_annotation=merge_via_annotation)
        com_model_obj.generate_community_model()

    if fixed_growth_rate is None:
        # Apply abundance (either None, "equal", or an abundance dict)
        if abundance == "equal":
            logger.info(f"Setting to equal abundance")
            com_model_obj.equal_abundance()
        elif isinstance(abundance, dict):
            logger.info(f"Setting abundances accordung to {abundance}")
            name_conversion = com_model_obj.generate_member_name_conversion_dict()
            tmp_abundance = {}
            try:
                for name, fraction in abundance.items():
                    tmp_abundance[name_conversion[name]] = fraction
            except KeyError as e:
                err_msg = f"Error: Some names in the abundances are not part of the model." \
                          f"\n\tAbundances: {abundance.keys()}" \
                          f"\n\tOrganisms in model: {com_model_obj.get_member_names()}"
                raise KeyError(err_msg)
            com_model_obj.convert_to_fixed_abundance()
            com_model_obj.apply_fixed_abundance(tmp_abundance)
        else:
            com_model_obj.convert_to_fixed_abundance()
    else:
        if fixed_growth_rate < 0.:
            logger.error(f"Error: Specified growth rate is negative ({fixed_growth_rate}). PyCoMo will continue with a "
                         f"growth rate set to 0.")
            fixed_growth_rate = 0.
        logger.info(f"Setting to fixed growth rate ({fixed_growth_rate})")
        com_model_obj.convert_to_fixed_growth_rate()
        com_model_obj.apply_fixed_growth_rate(fixed_growth_rate)

    # Apply medium
    if medium is not None:
        logger.info(f"Loading medium from file")
        com_model_obj.load_medium_from_file(medium)
        com_model_obj.apply_medium()

    if sbml_output_file is not None:
        logger.info(f"Saving model to {os.path.join(out_dir, sbml_output_file)}")
        com_model_obj.save(os.path.join(out_dir, sbml_output_file))

    fva_flux_vector = None
    if fva_interaction_file is not None:
        logger.info(f"Running FVA")
        try:
            interaction_df, fva_flux_vector = com_model_obj.potential_metabolite_exchanges(
                fba=False,
                composition_agnostic=composition_agnostic,
                fraction_of_optimum=fva_solution_threshold,
                loopless=loopless,
                processes=num_cores,
                return_flux_vector=True)
            logger.info(f"Saving flux vector to {os.path.join(out_dir, fva_interaction_file)}")
            interaction_df.to_csv(os.path.join(out_dir, fva_interaction_file), sep="\t", header=True,
                                  index=False, float_format='%f')
        except cobra.exceptions.Infeasible:
            logger.warning(f"FVA of community is infeasible. No FVA interaction file was generated.")

    if fva_solution_file is not None:
        logger.info(f"Running FVA")
        try:
            if fva_flux_vector is not None:
                # Use flux vector from interactions
                fva_flux_vector.to_csv(os.path.join(out_dir, fva_solution_file), sep="\t", header=True,
                                       index=False, float_format='%f')
            else:
                com_model_obj.fva_solution_flux_vector(file_path=os.path.join(out_dir, fva_solution_file),
                                                       fraction_of_optimum=fva_solution_threshold,
                                                       processes=num_cores,
                                                       composition_agnostic=composition_agnostic,
                                                       loopless=loopless)
        except cobra.exceptions.Infeasible:
            logger.warning(f"FVA of community is infeasible. No FVA flux vector file was generated.")

    fba_flux_vector = None
    if fba_interaction_file is not None:
        logger.info(f"Running FBA")
        try:
            interaction_df, fba_flux_vector = com_model_obj.potential_metabolite_exchanges(fba=True,
                                                                                           return_flux_vector=True)
            logger.info(f"Saving flux vector to {os.path.join(out_dir, fba_interaction_file)}")
            interaction_df.to_csv(os.path.join(out_dir, fba_interaction_file), sep="\t", header=True,
                                  index=False, float_format='%f')
        except cobra.exceptions.Infeasible:
            logger.warning(f"FBA of community is infeasible. No FBA interaction file was generated.")

    if fba_solution_file is not None:
        logger.info(f"Running FBA")
        try:
            if fba_flux_vector is not None:
                # Use flux vector from interactions
                fva_flux_vector.to_csv(os.path.join(out_dir, fba_solution_file), sep="\t", header=True,
                                       index=False, float_format='%f')
            else:
                com_model_obj.fba_solution_flux_vector(file_path=os.path.join(out_dir, fba_solution_file))
        except cobra.exceptions.Infeasible:
            logger.warning(f"FBA of community is infeasible. No FBA flux vector file was generated.")

    if max_growth_rate_file is not None:
        logger.info(f"Calculating the maximum growth rate")
        try:
            growth_df = com_model_obj.max_growth_rate(sensitivity=4, return_abundances=True)
            growth_df.to_csv(os.path.join(out_dir, max_growth_rate_file))
        except cobra.exceptions.Infeasible:
            logger.warning(f"FBA of community is infeasible. No maximum growth-rate file was generated.")

    if return_as_cobra_model:
        # Retrieve community model
        return com_model_obj.model
    else:
        return com_model_obj  # Return the community model object


def main():
    """
    The main function to be executed when PyCoMo is used via the command line.
    """
    parser = create_arg_parser()
    args = parser.parse_args()
    args = check_args(args)
    if args.log_file is not None:
        configure_logger(log_file=args.log_file)
    if args.log_level is not None:
        configure_logger(level=args.log_level)
    logger.info(args)
    if args.abundance is not None and args.abundance != "equal":
        # Retrieve the abundance from file
        args.abundance = read_abundance_from_file(args.abundance)

    if args.is_community:
        doall(com_model=args.input[0],
              community_name=args.name,
              out_dir=args.output_dir,
              abundance=args.abundance,
              medium=args.medium,
              fba_solution_file=args.fba_solution_file,
              fva_solution_file=args.fva_solution_file,
              fva_solution_threshold=args.fraction_of_optimum,
              fba_interaction_file=args.fba_interaction_file,
              fva_interaction_file=args.fva_interaction_file,
              sbml_output_file=args.sbml_output_file,
              return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation,
              num_cores=args.num_cores,
              composition_agnostic=args.composition_agnostic,
              loopless=args.loopless,
              max_growth_rate_file=args.max_growth_rate_file)

    elif len(args.input) == 1 and os.path.isdir(args.input[0]):
        doall(model_folder=args.input[0],
              community_name=args.name,
              out_dir=args.output_dir,
              abundance=args.abundance,
              medium=args.medium,
              fba_solution_file=args.fba_solution_file,
              fva_solution_file=args.fva_solution_file,
              fva_solution_threshold=args.fraction_of_optimum,
              fba_interaction_file=args.fba_interaction_file,
              fva_interaction_file=args.fva_interaction_file,
              sbml_output_file=args.sbml_output_file,
              return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation,
              num_cores=args.num_cores,
              composition_agnostic=args.composition_agnostic,
              loopless=args.loopless,
              max_growth_rate_file=args.max_growth_rate_file)
    else:
        doall(models=args.input,
              community_name=args.name,
              out_dir=args.output_dir,
              abundance=args.abundance,
              medium=args.medium,
              fba_solution_file=args.fba_solution_file,
              fva_solution_file=args.fva_solution_file,
              fva_solution_threshold=args.fraction_of_optimum,
              fba_interaction_file=args.fba_interaction_file,
              fva_interaction_file=args.fva_interaction_file,
              sbml_output_file=args.sbml_output_file,
              return_as_cobra_model=False,
              merge_via_annotation=args.match_via_annotation,
              num_cores=args.num_cores,
              composition_agnostic=args.composition_agnostic,
              loopless=args.loopless,
              max_growth_rate_file=args.max_growth_rate_file)

    logger.info("All done!")
    sys.exit(0)


if __name__ == "__main__":
    # get the path of this script and add it to the "pythonpath"
    SCRIPT_PATH = os.path.split(os.path.realpath(os.path.abspath(__file__)))[0]
    sys.path.insert(0, SCRIPT_PATH)

    main()
