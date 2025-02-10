import pytest
import logging
import multiprocessing
from unittest.mock import patch
import sys
import os
import cobra
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pycomo
from pycomo.helper.multiprocess import loopless_fva, _loopless_fva_step

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

def dummy_com_model():
    model = dummy_model()
    model.objective = "EX_bio"
    com_model = pycomo.CommunityModel(models=[
        pycomo.SingleOrganismModel(model.copy(), name="m1"),
        pycomo.SingleOrganismModel(model.copy(), name="m2")]
    )
    
    return com_model

# Define a mock function that always times out
def mock_fva_step(_):
    raise multiprocessing.TimeoutError("Mock timeout")

def mock_fva_step_sleep(r):
    try:
        print("Starting to sleep")
        time.sleep(0.2)
        print("Finished sleeping")
        return r, 1., 0.
    except Exception as e:
        return e

@pytest.mark.asyncio
async def test_fva_worker_timeout_handling(caplog):
    """
    Test whether the FVA function correctly retries a task and logs a timeout when a worker takes too long.
    """

    # Store the original function so we can restore it later
    original_fva_step = _loopless_fva_step

    try:
        # Monkey-patch the function globally
        pycomo.helper.multiprocess._loopless_fva_step = mock_fva_step_sleep

        if os.path.isfile("test/data/output/multiprocess.log"):
            os.remove("test/data/output/multiprocess.log")
        pycomo.configure_logger(level="info", log_file="test/data/output/multiprocess.log")

        model = dummy_com_model()
        result = loopless_fva(pycomo_model=model,
                                reactions=model.model.reactions,
                                processes=2,
                                time_out=0.1,
                                max_time_out=0.2)  # Ensure multiprocessing is used

        # Check if the timeout warning was logged
        with open("test/data/output/multiprocess.log", "r") as log_file:
            assert any("FVA step timed out" in message for message in log_file.readlines()), "Timeout warning not logged!"

    finally:
        # Restore the original function to avoid side effects
        pycomo.helper.multiprocess._loopless_fva_step = original_fva_step


@pytest.mark.asyncio
async def test_fva_worker_timeout_repetition():
    """
    Test whether the FVA function correctly retries a task and logs a timeout when a worker takes too long.
    """

    # Store the original function so we can restore it later
    original_fva_step = _loopless_fva_step

    try:
        # Monkey-patch the function globally
        pycomo.helper.multiprocess._loopless_fva_step = mock_fva_step_sleep

        if os.path.isfile("test/data/output/multiprocess.log"):
            os.remove("test/data/output/multiprocess.log")
        pycomo.configure_logger(level="debug", log_file="test/data/output/multiprocess.log")

        model = dummy_com_model()
        result = loopless_fva(pycomo_model=model,
                                reactions=model.model.reactions,
                                processes=2,
                                time_out=0.1,
                                max_time_out=1.)

        # Check if the timeout warning was logged
        with open("test/data/output/multiprocess.log", "r") as log_file:
            lines = log_file.readlines()
            assert any("FVA step timed out" in message for message in lines), "Timeout warning not logged!"
            assert any("Repeating failed FVA steps for reactions" in message for message in lines), "Repetition info not logged!"
            assert any("Processed 100.0% of fva steps" in message for message in lines), "Not all reactions finished!"
        assert all(result["maximum"]==1.)
        assert all(result["minimum"]==0.)
            

    finally:
        # Restore the original function to avoid side effects
        pycomo.helper.multiprocess._loopless_fva_step = original_fva_step

