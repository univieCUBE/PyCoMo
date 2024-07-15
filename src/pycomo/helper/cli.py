import os
import sys
import argparse

__description__ = 'A package for generating community metabolic models from single species/strain models.'
__author__ = 'Michael Predl, Marianne Mießkes'
__license__ = "MIT"
__version__ = "0.2.1"


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
                                  help="set abundances to be equal for all community members")

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

    pg_output.add_argument('--fba-flux', action='store_true',
                           help="run FBA on the community model and store the flux vector in a file")

    pg_output.add_argument('--fva-flux', type=float,
                           help="run FVA on the exchange metabolites of the community model and store the flux vector "
                                "in a file. Set the threshold of the objective that needs to be achieved.")

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

    if args.fva_flux is not None:
        if not 0. <= args.fva_flux <= 1.:
            raise ValueError("The fva-flux argument needs to be between 0. and 1. (inclusive).")

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

    args.fba_solution_path = None
    if args.fba_flux:
        args.fba_solution_path = os.path.join(args.output_dir, f"{args.name}_fba_flux.csv")

    args.fva_solution_path = None
    if args.fva_flux is not None:
        args.fva_solution_path = os.path.join(args.output_dir, f"{args.name}_fva_flux.csv")

    args.fba_interaction_path = None
    if args.fba_interaction:
        args.fba_interaction_path = os.path.join(args.output_dir, f"{args.name}_fba_interaction_flux.csv")

    args.fva_interaction_path = None
    if args.fva_interaction:
        args.fva_interaction_path = os.path.join(args.output_dir, f"{args.name}_fva_interaction_flux.csv")

    args.sbml_output_path = os.path.join(args.output_dir, f"{args.name}.xml")

    return args
