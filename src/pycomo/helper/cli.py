import os
import sys
import argparse
import warnings

__description__ = 'A package for generating community metabolic models from single species/strain models.'
__author__ = 'Michael Predl, Marianne Mießkes'
__license__ = "MIT"
__version__ = "0.2.7"


def parse_fva_flux(value):
    if value is None:  # Used as a flag (e.g., --fva-flux)
        return True
    try:
        # Try to convert the value to a float (e.g., --fva-flux 0.5)
        float_value = float(value)
        # Issue a deprecation warning
        warnings.warn(
            "Providing a float value for '--fva-flux' is deprecated and will be removed in a future version. "
            "Please use it as a flag instead and set the fraction of the optimum with '--fraction-of-optimum'",
            DeprecationWarning,
        )
        return float_value
    except ValueError:
        # Raise an error if the value is invalid
        raise argparse.ArgumentTypeError(f"Invalid value for --fva-flux: {value}")


def create_arg_parser():
    """
    Creates a parser object for the command line interface

    :return: An argparse argument parser
    """
    parser = argparse.ArgumentParser(prog="PyCoMo")

    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}',
                        help="display PyCoMo version")

    parser.add_argument('-i', '--input', nargs='+', type=str, required=True,
                        help="single species/strain models to combine, either as a directory or separate files")

    parser.add_argument('-c', '--is-community', action='store_true',
                        help="Set this flag if the input model is already a community model.")

    # All parameters regarding the generation and contextualisation of the community model
    pg_com_model = parser.add_argument_group('Community model parameters')

    pg_com_model.add_argument('-n', '--name', type=str, default='community_model',
                              help="the name for the new community model")

    pg_com_model.add_argument('-m', '--match-via-annotation', type=str,
                              help="the metabolite annotation type to use for matching exchange metabolites of "
                                   "different community members (e.g. metanetx.chemical)")

    pg_linearisation = pg_com_model.add_mutually_exclusive_group()

    pg_linearisation.add_argument('--growth-rate', type=float,
                                  help="set a fixed growth-rate for the community")

    pg_linearisation.add_argument('--equal-abd', action='store_true',
                                  help="set abundances to be equal for all community members")

    pg_linearisation.add_argument('--abd-file', type=str,
                                  help="a comma separated file containing the input model file names and their "
                                       "abundance. No header should be used in the file.")

    pg_com_model.add_argument('--medium', type=str,
                              help="the medium to be used in the community model, as a comma separated file "
                                   "containing a column 'compounds' and a column 'maxFlux'.")

    pg_com_model.add_argument('--num-cores', type=int,
                              help="the number of cores to be used for FVA")

    # All parameters regarding outputs to be produced
    pg_output = parser.add_argument_group('Output parameters')

    pg_output.add_argument('-o', '--output-dir', default=os.getcwd(), type=str,
                           help="the output directory to store results (default is the current working directory)")

    pg_output.add_argument('--save-sbml', action='store_true', default=False,
                           help="save the community metabolic model as sbml file")

    pg_output.add_argument('--fba-flux', action='store_true',
                           help="run FBA on the community model and store the flux vector in a file")

    pg_output.add_argument('--fva-flux', nargs='?', type=parse_fva_flux, const=True, default=False,
                           help="run FVA on the exchange metabolites of the community model and store the flux vector "
                                "in a file.")

    pg_output.add_argument('--fba-interaction', action='store_true',
                           help="run FBA on the community model and store the flux of exchange metabolites and "
                                "whether they are cross-fed in a file")

    pg_output.add_argument('--fva-interaction', action='store_true',
                           help="run FVA on the community model and store the flux of exchange metabolites and "
                                "whether they are cross-fed in a file. Set the threshold of the objective that needs "
                                "to be achieved.")

    pg_output.add_argument('--composition-agnostic', action='store_true', default=False,
                           help="run FVA with relaxed constraints, to calculate all possible cross-feeding "
                                "interactions across all community growth-rates and abundance profiles.")

    pg_output.add_argument('--loopless', type=bool, default=True,
                           help="run FVA with loop correction (on by default)")

    pg_output.add_argument('--fraction-of-optimum', type=float,
                           help="set the fraction of optimum that needs to be achieved. Values need to be between 0 "
                                "and 1. Examples: 0 -> 0%% of optimum, 0.9 -> 90%% of optimum, 1 -> 100%% of optimum.")

    pg_output.add_argument('--max-growth-rate', action='store_true',
                           help="calculate the maximum growth-rate of the community, as well as the community "
                                "composition reaching it. Results are stored in a csv file.")

    pg_output.add_argument('--log-file', nargs='?', type=str, const="pycomo.log", default=None,
                           help="set a log file for PyCoMo, located in the output directory (see -o / --output-dir. "
                                "If used as flag, the file is called 'pycomo.log'.")

    pg_output.add_argument('--log-level', type=str,
                           help="set log level. Use one of the following values: error, warning, info, debug")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser


def check_args(args):
    """
    Check whether all arguments are in correct format.

    :param args:
    :raises ValueError: This error is raised if one of the supplied arguments is not in the correct format or the
        specified range of values
    :return: parsed arguments
    """
    if not os.path.isdir(args.output_dir):
        raise ValueError("The output-dir does not exist or is not a directory.")
    if args.input is None:
        raise ValueError("Please provide input to run PyCoMo (-i / --input)")
    elif not all([os.path.exists(arg_path) for arg_path in args.input]):
        raise ValueError("Not all input files / directories exist.")

    if args.fraction_of_optimum is not None:
        if not 0. <= args.fraction_of_optimum <= 1.:
            raise ValueError("The fraction-of-optimum argument needs to be between 0. and 1. (inclusive).")

    if args.num_cores is not None:
        if not 0 < args.num_cores:
            raise ValueError("The number of cores must be greater than 0.")

    if args.abd_file is not None and not os.path.isfile(args.abd_file):
        raise ValueError("The abundance file is not a file or does not exist")

    if args.growth_rate is not None and args.growth_rate < 0.:
        raise ValueError(f"The specified growth rate ({args.growth_rate}) is negative.")

    if args.medium is not None and not os.path.isfile(args.medium):
        raise ValueError("The medium file is not a file or does not exist")

    args.abundance = None
    if args.equal_abd:
        args.abundance = "equal"
    elif args.abd_file is not None:
        args.abundance = args.abd_file

    args.fba_solution_file = None
    if args.fba_flux:
        args.fba_solution_file = f"{args.name}_fba_flux.csv"

    args.fva_solution_file = None
    if args.fva_flux:
        args.fva_solution_file = f"{args.name}_fva_flux.csv"

    args.fba_interaction_file = None
    if args.fba_interaction:
        args.fba_interaction_file = f"{args.name}_fba_interaction_flux.csv"

    args.fva_interaction_file = None
    if args.fva_interaction:
        args.fva_interaction_file = f"{args.name}_fva_interaction_flux.csv"

    args.sbml_output_file = None
    if args.save_sbml:
        args.sbml_output_file = f"{args.name}.xml"

    args.max_growth_rate_file = None
    if args.max_growth_rate:
        args.max_growth_rate_file = f"{args.name}_max_growth_rate.csv"

    if args.log_file is not None:
        args.log_file = os.path.join(args.output_dir, args.log_file)

    if args.log_level is not None:
        if args.log_level.lower() not in ["debug", "info", "warning", "error"]:
            warnings.warn(
                f"Unknown log-level {args.log_level}. Please use one of the following: error, warning, info, debug"
            )
            args.log_level = None

    return args
