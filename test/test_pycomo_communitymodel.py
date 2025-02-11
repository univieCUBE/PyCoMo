import sys
from pathlib import Path
import os

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pycomo
import cobra
import pytest

community_input = "test/data/koch_com_model.xml"
community_output = "test/data/output/koch_com_model.xml"
flux_output = "test/data/gut_flux.csv"
toy_folder = "data/toy/gut"


def test_run_fva_mu_c():
    # Test if mu_c is set correctly for fva
    target_mu_c = 0.01
    toy_model = pycomo.CommunityModel.load(community_input)
    toy_model.convert_to_fixed_growth_rate(0.02)
    original_mu_c = toy_model.mu_c
    solution = toy_model.run_fva(fva_mu_c=target_mu_c)
    biomass_flux = solution.loc["community_biomass", :]

    # Check results of mu_c with model in fixed growth mode
    assert biomass_flux["min_flux"] == target_mu_c
    assert biomass_flux["max_flux"] == target_mu_c

    # Check that mu_c of the CommunityModel remains unchanged
    assert toy_model.mu_c == original_mu_c

    # Check results of mu_c with model in fixed abundance mode
    toy_model = pycomo.CommunityModel.load(community_input)
    toy_model.convert_to_fixed_abundance()
    solution = toy_model.run_fva(fva_mu_c=target_mu_c)
    biomass_flux = solution.loc["community_biomass", :]
    assert biomass_flux["min_flux"] == target_mu_c
    assert biomass_flux["max_flux"] == target_mu_c


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


def test_community_model_boundary_biomass_rxn():
    model = dummy_model()
    model.objective = "EX_bio"
    com_model = pycomo.CommunityModel(models=[
        pycomo.SingleOrganismModel(model.copy(), name="m1"),
        pycomo.SingleOrganismModel(model.copy(), name="m2")]
    )
    com_model.convert_to_fixed_abundance()
    sol = com_model.model.optimize()
    assert sol.status == "optimal"
    assert sol.objective_value == 1000.


def test_community_model_special_rxn_getters():
    transfer_sol = {'m1_TF_a_m1_external',
                    'm1_to_community_biomass',
                    'm2_TF_a_m2_external',
                    'm2_to_community_biomass'}
    transport_sol = {'m1_TP_a', 'm1_bio', 'm2_TP_a', 'm2_bio'}
    f_rxn_sol = {'SK_m1_TF_a_m1_external_lb',
                 'SK_m1_TF_a_m1_external_ub',
                 'SK_m1_TP_a_ub',
                 'SK_m1_bio_ub',
                 'SK_m1_to_community_biomass_ub',
                 'SK_m2_TF_a_m2_external_lb',
                 'SK_m2_TF_a_m2_external_ub',
                 'SK_m2_TP_a_ub',
                 'SK_m2_bio_ub',
                 'SK_m2_to_community_biomass_ub',
                 'm1_fraction_reaction',
                 'm2_fraction_reaction'}
    model = dummy_model()
    model.objective = "bio"
    com_model = pycomo.CommunityModel(models=[
        pycomo.SingleOrganismModel(model.copy(), name="m1"),
        pycomo.SingleOrganismModel(model.copy(), name="m2")]
    )
    transfer_rxns = set([r.id for r in com_model.transfer_reactions])
    transport_rxns = set([r.id for r in com_model.transport_reactions])
    f_rxns = set([r.id for r in com_model.f_reactions])
    assert transfer_rxns == transfer_sol
    assert transport_rxns == transport_sol
    assert f_rxns == f_rxn_sol


def test_community_model_low_abundance_warning():
    if os.path.isfile("test/data/output/community_model.log"):
        os.remove("test/data/output/community_model.log")
    pycomo.configure_logger(level="info", log_file="test/data/output/community_model.log")

    model = dummy_model()
    model.objective = "EX_bio"
    com_model = pycomo.CommunityModel(models=[
        pycomo.SingleOrganismModel(model.copy(), name="m1"),
        pycomo.SingleOrganismModel(model.copy(), name="m2")]
    )
    com_model.convert_to_fixed_abundance()
    com_model.apply_fixed_abundance({"m1": 10**-10, "m2": 1-(10**-10)})  # Assign very low abundance
    
    with open("test/data/output/community_model.log", "r") as log_file:
        lines = log_file.readlines()
        assert any("Abundance of m1 is lower than the solver tolerance" in message for message in lines), "Abundance warning not logged!"
    
    cobra.Configuration().tolerance = 10**-1
    com_model.apply_fixed_abundance({"m2": 10**-2, "m1": 1-(10**-2)})  # Assign very low abundance
    
    with open("test/data/output/community_model.log", "r") as log_file:
        lines = log_file.readlines()
        assert any("Abundance of m2 is lower than the solver tolerance" in message for message in lines), "Abundance warning not logged!"
    
    cobra.Configuration().tolerance = 10**-6
