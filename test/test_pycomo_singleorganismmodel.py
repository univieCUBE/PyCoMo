import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pycomo
import cobra
import pytest


def test_single_organism_model_empty():
    with pytest.raises(TypeError):
        test_model = pycomo.SingleOrganismModel()


def dummy_model():
    """
    Create a very simple dummy model for testing.
    Model: -> a_ex -> a -> bio ->

    :return: Dummy Model
    """
    model = cobra.Model()
    r_bio = cobra.Reaction("bio")
    r_tp = cobra.Reaction("TP_a")
    model.add_metabolites([cobra.Metabolite(i) for i in ["a", "a_external", "bio"]])
    model.add_reactions([r_bio, r_tp])
    r_bio.add_metabolites({"a": -1, "bio": 1})
    model.metabolites.get_by_id("bio").compartment = "external"
    model.metabolites.get_by_id("a").compartment = "internal"
    model.metabolites.get_by_id("a_external").compartment = "external"
    model.add_boundary(model.metabolites.get_by_id("a_external"), type="exchange")
    model.add_boundary(model.metabolites.get_by_id("bio"), type="exchange")
    r_tp.add_metabolites({"a_external": -1, "a": 1})
    return model


def test_prepare_for_merging_boundary_biomass_rxn():
    model = dummy_model()
    model.objective = "EX_bio"
    merge_model = pycomo.SingleOrganismModel(model, "name").prepare_for_merging()
    # Biomass metabolite is moved from boundary compartment
    assert merge_model.metabolites.get_by_id("name_bio").compartment == "name_bio"
    # Biomass metabolite does not have boundary reactions
    assert not any([r in model.boundary for r in merge_model.metabolites.get_by_id("name_bio").reactions])
