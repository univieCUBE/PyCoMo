import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pycomo
import cobra
import os

community_output = "test/data/output/gut_community.xml"
toy_folder = "data/toy/gut"
toy_folder_tiny = "data/use_case/toy_models/"


def test_doall():
    community_model = pycomo.doall(toy_folder)
    # Compare the output with the reference
    ref_model = cobra.io.read_sbml_model(community_output)
    assert len(community_model.model.metabolites) == len(ref_model.metabolites)
    assert len(community_model.model.reactions) == len(ref_model.reactions)
    assert len(community_model.model.groups) == len(ref_model.groups)


def test_doall_max_growth():
    community_model = pycomo.doall(toy_folder_tiny, max_growth_rate_file="test/data/output/max_growth_rate.csv")
    df = pd.read_csv("test/data/output/max_growth_rate.csv")
    assert all(df[df["reaction_id"] == "community_biomass"]["min_flux"] >= 14.9999)
    assert all(df[df["reaction_id"] == "community_biomass"]["max_flux"] >= 14.9999)
    assert all(df[df["reaction_id"] == "toy1_fraction_reaction"]["max_flux"] == 0.)
    assert all(df[df["reaction_id"] == "toy2_2_fraction_reaction"]["min_flux"] == 1.)


def test_doall_fva_interaction_and_flux_vector(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['pycomo',
                                '-i', 'data/use_case/toy_models',
                                '-o', 'test/data/output/doall_fva',
                                '--fva-flux',
                                '--fva-interaction',
                                '--fraction-of-optimum', '0.8',
                                '--loopless', 'True'
                                ])
        try:
            pycomo.pycomo_models.main()
        except SystemExit as e:
            print(e)
            if e is None or int(str(e)) == 0:
                pass  # pycomo finished and exited
            else:
                raise SystemExit(e)

        df = pd.read_csv("test/data/output/doall_fva/community_model_fva_flux.csv", sep="\t")
        assert all(df[df["reaction_id"] == "community_biomass"]["max_flux"] == 10.)
        assert all(df[df["reaction_id"] == "community_biomass"]["min_flux"] == 8.)


def test_doall_set_log(monkeypatch):
    if os.path.isfile('test/data/output/doall_fva/test.log'):
        os.remove('test/data/output/doall_fva/test.log')

    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['pycomo',
                                '-i', 'data/use_case/toy_models',
                                '-o', 'test/data/output/doall_fva',
                                '--log-level', 'error',
                                '--log-file', 'test.log'
                                ])
        try:
            pycomo.pycomo_models.main()
        except SystemExit as e:
            print(e)
            if e is None or int(str(e)) == 0:
                pass  # pycomo finished and exited
            else:
                raise SystemExit(e)

        assert os.path.isfile('test/data/output/doall_fva/test.log')

