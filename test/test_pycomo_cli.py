import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import pycomo
import cobra
import pytest
import argparse
import warnings

community_output = "test/data/output/gut_community.xml"
toy_folder = "data/toy/gut"
toy_folder_tiny = "data/use_case/toy_models/"


def test_fva_flux_float_with_warning(monkeypatch):
    with monkeypatch.context() as m:
        with pytest.warns(DeprecationWarning, match="Providing a float value for '--fva-flux' is deprecated"):
            m.setattr(sys, 'argv', ['pycomo',
                                    '-i', 'data/use_case/toy_models',
                                    '-o', 'test/data/output/doall_fva',
                                    '--fva-flux', '0.5',
                                    '--fva-interaction',
                                    '--fraction-of-optimum', '0.8',
                                    '--loopless', 'True'
                                    ])
            parser = pycomo.helper.cli.create_arg_parser()
            args = parser.parse_args()
            assert args.fva_flux == 0.5  # Should correctly parse the float value


def test_fva_flux_default(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['pycomo',
                                '-i', 'data/use_case/toy_models',
                                '-o', 'test/data/output/doall_fva',
                                '--fva-interaction',
                                '--fraction-of-optimum', '0.8',
                                '--loopless', 'True'
                                ])
        parser = pycomo.helper.cli.create_arg_parser()
        args = parser.parse_args()
        assert args.fva_flux is False  # Should be False by default


def test_fva_flux_invalid_value(monkeypatch):
    with monkeypatch.context() as m:
        # Expect an error when an invalid value is passed
        with pytest.raises(SystemExit):
            with pytest.raises(argparse.ArgumentError,
                               match="argument --fva-flux: Invalid value for --fva-flux: invalid"):
                m.setattr(sys, 'argv', ['pycomo',
                                        '-i', 'data/use_case/toy_models',
                                        '-o', 'test/data/doall_fva',
                                        '--fva-interaction',
                                        '--fva-flux', 'invalid',
                                        '--fraction-of-optimum', '0.8',
                                        '--loopless', 'True'
                                        ])
                parser = pycomo.helper.cli.create_arg_parser()
                args = parser.parse_args()


def test_cli_log_level_invalid(monkeypatch):
    with monkeypatch.context() as m:
        with pytest.warns(Warning,
                          match=f"Unknown log-level invalid. Please use one of the following: "
                                f"error, warning, info, debug"):
            m.setattr(sys, 'argv', ['pycomo',
                                    '-i', 'data/use_case/toy_models',
                                    '-o', 'test/data/output/doall_fva',
                                    '--fva-flux', '0.5',
                                    '--fva-interaction',
                                    '--fraction-of-optimum', '0.8',
                                    '--loopless', 'True',
                                    '--log-level', 'invalid'
                                    ])
            parser = pycomo.helper.cli.create_arg_parser()
            args = parser.parse_args()
            args = pycomo.helper.cli.check_args(args)
            assert args.log_level is None  # Should change to None


def test_cli_log_file(monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sys, 'argv', ['pycomo',
                                '-i', 'data/use_case/toy_models',
                                '-o', 'test/data/output/doall_fva',
                                '--fva-flux', '0.5',
                                '--fva-interaction',
                                '--fraction-of-optimum', '0.8',
                                '--loopless', 'True',
                                '--log-file', 'test.log'
                                ])
        parser = pycomo.helper.cli.create_arg_parser()
        args = parser.parse_args()
        args = pycomo.helper.cli.check_args(args)
        assert 'test.log' in args.log_file
