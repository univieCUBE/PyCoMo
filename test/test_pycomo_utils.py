import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pycomo
from pycomo.helper import utils
import cobra
import pytest


model = cobra.io.read_sbml_model("data/toy/gut/F_plautii_YL31.xml")


def test_get_model_biomass_compound_wrong_id():
    with pytest.raises(KeyError):
        utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", expected_biomass_id="a", generate_if_none=True)


def test_get_model_biomass_compound_correct_id():
    biomass_met_id = "cpd11416_c0"
    biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", expected_biomass_id=biomass_met_id, generate_if_none=True)
    assert biomass_met.id == biomass_met_id


def test_get_model_biomass_compound_no_id_no_generate():
    with pytest.raises(KeyError):
        biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", generate_if_none=False)


def test_get_model_biomass_compound_no_id_generate():
    biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", generate_if_none=True)
    assert biomass_met.id == "cpd11416_shared_comp_name"


def test_get_model_biomass_compound_multiple_products_no_generate():
    with model as multi_model:
        multi_model.objective = "rxn00006_c0"
        with pytest.raises(KeyError):
            biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", generate_if_none=False)


def test_get_model_biomass_compound_multiple_products_generate():
    with model as multi_model:
        multi_model.objective = "rxn00006_c0"
        biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", generate_if_none=True)
        assert biomass_met.id == "cpd11416_shared_comp_name"


def test_get_model_biomass_compound_single_product():
    with model as single_model:
        single_model.objective = "rxn00022_c0"
        biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", generate_if_none=True)
        assert biomass_met.id == "cpd00027_c0"


def test_get_model_biomass_compound_expected_in_objective():
    with model as multi_model:
        multi_model.objective = "bio1"
        biomass_met = utils.get_model_biomass_compound(model, shared_compartment_name="shared_comp_name", expected_biomass_id="cpd11416_c0", generate_if_none=True)
        assert biomass_met.id == "cpd11416_c0"



