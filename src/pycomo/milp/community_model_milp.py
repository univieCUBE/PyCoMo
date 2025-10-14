from pycomo.pycomo_models import CommunityModel
from pycomo.helper.utils import replace_metabolite_stoichiometry
from pycomo.milp.milp_helper import mu_approx_resolution, make_mu_bins, petersen_linearization

from sympy import Add


OPTLANG_BINARY = 'binary'

class CommunityModelMILP(CommunityModel):
    """
    Community model class with additional functionalities, which depend on MILP solvers.
    Functionalities include: Binarized growth rate, exclusion of thermodynamically infeasible loops.
    """

    mu_lb = 0.
    mu_ub = None
    n_mu_bins = None
    mu_vars = []
    ga_vars = []


    def relax_model_linearisation(self):
        """
        Remove model linearisation constraints (fixed growth rate, fixed abundance).

        """
        self.equal_abundance()
        member_names = self.get_member_names()
        # Relax fraction reaction
        for m in member_names:
            fraction_rxn = self.model.reactions.get_by_id(f"{m}_fraction_reaction")
            try:
                fraction_met = self.model.metabolites.get_by_id(f"{m}_f_biomass_met")
            except KeyError:
                fraction_met = self._backup_metabolites[f"{m}_f_biomass_met"]
                self.model.add_metabolites(fraction_met)
            fraction_rxn.bounds = (0., self.max_flux)
            replace_metabolite_stoichiometry(fraction_rxn, {fraction_met: 0.})
            bio2com_rxn = self.model.reactions.get_by_id(f"{m}_to_community_biomass")
            replace_metabolite_stoichiometry(bio2com_rxn, {fraction_met: 0.})
        # Remove equal growth (at fixed abundance)
        self.model.reactions.abundance_reaction.bounds = 0., 0.

        self.fixed_abundance_flag = False
        self.fixed_growth_rate_flag = False
        return    

    def init_f_variables(self, track_vars=None):
        f_vars = {}
        for member in self.get_member_names():
            f_var = self.model.solver.interface.Variable(f"{member}_fraction_variable")
            f_vars[member] = f_var
            cons_f = self.model.solver.interface.Constraint(
                self.model.reactions.get_by_id(f"{member}_fraction_reaction").flux_expression - 1. * f_var, lb=0, ub=0, 
                name=f"{member}_fraction_constraint"
            )
            if track_vars is not None:
                track_vars["cons"].append(cons_f)
                track_vars["vars"].append(f_var)
            self.model.solver.add(f_var, cons_f)
        return f_vars

    def init_mu_variables(self, model, r_bio, mu_lb, mu_ub, n_mu_bins, track_vars=None):
        """
        Necessary for the zeroth order approximation of mu:

        .. math::

            mu \in [0.1, 0.9] , nbins = 8
            => mu = 0.15 OR mu = 0.25 OR ... OR mu = 0.85

        Using binary expansion of the bins instead of a list of 0-1s
        described `here <https://orinanobworld.blogspot.ch/2013/07/integer-variables-and-quadratic-terms.html>`_

        :return:
        """

        mu_bins = make_mu_bins(mu_lb, mu_ub, n_mu_bins)
        ga = list()
        N = n_mu_bins
        n_vars = int(np.ceil(np.log2(N)))

        for e in range(n_vars):
            ga_i = model.solver.interface.Variable(kind="GA",
                                        hook=model,
                                        id_=str(2 ** e),
                                        name=str(2 ** e), type="binary")
            model.solver.add([ga_i])
            if track_vars is not None:
                track_vars["vars"].append(ga_i)
            ga.append(ga_i)

        # Force that only one growth range can be chosen:
        # b0*2^0 + b1*2^1 + b2*2^2 + ... + bn*2^n <= n_bins
        # this constraint always holds, so I removed it
        # choice_expr = sum(ga)
        # self.add_constraint(kind=GrowthChoice,
        #                     hook=self,
        #                     expr=choice_expr,
        #                     id_='growth',
        #                     ub=self.n_mu_bins,
        #                     lb=0)

        # Couple growth

        # |v_net - mu| <= bin_width
        bin_half_width = max([(x[1] - x[0]) / 2 for _, x in mu_bins])

        the_integer = Add(*[(2 ** i) * ga_i for i, ga_i in enumerate(ga)])

        binarized_mu = mu_lb + the_integer * mu_approx_resolution(mu_ub, mu_lb, n_mu_bins)

        growth_coupling_expr = r_bio.flux_expression - binarized_mu

        cons_bin = model.problem.Constraint(expression=growth_coupling_expr, name="GA_c_bin",
                            ub=bin_half_width,
                            lb=-1 * bin_half_width)

        # This one is probably superfluous
        #cons_bin_mu_bound = model.problem.Constraint(expression=binarized_mu, name="GA_c_bin_bound",
        #                    ub=mu_ub,
        #                    lb=mu_lb)

        cons_mu_bound = model.problem.Constraint(expression=r_bio.flux_expression, name="GA_c_mu_bound",
                            ub=mu_ub,
                            lb=mu_lb)
        if track_vars is not None:
            track_vars["cons"] += [cons_bin, cons_mu_bound]
        model.solver.add([cons_bin, cons_mu_bound])


        # So that the solver spends less time looking for an ub on the objective
        # when optimizing for growth
        #r_bio.upper_bound = mu_ub + mu_approx_resolution(mu_ub, mu_lb, n_mu_bins)

        # Update the variable indices
        #self.regenerate_variables()
        #self.regenerate_constraints()
        return ga

    def linearize_me(self, model, r_bio, f_var, ga_vars, mu_ub, mu_lb, n_mu_bins, queue=False, track_vars=None):
        """
        Performs Petersen linearization on μ*E to keep a MILP problem

        :return:
        """

        E = f_var

        # z = lambda_i * E_hat <= big_M
        big_m = 1.

        # ga_i is a binary variable for the binary expansion f the fraction on N
        # of the max growth rate
        ga_vars = self.get_ordered_ga_vars(ga_vars)

        out_expr = - r_bio.flux_expression

        # Build z =   ga_0*2^0*mu_max/N * [E]
        #           + ga_1*2^1*mu_max/N * [E]
        #           + ...
        #           + ga_n*2^n*mu_max/N * [E]

        for i, ga_i in enumerate(ga_vars):
            # Linearization step for ga_i * [E]
            z_name = '__MUL__'.join([ga_i.name, E.name])
            # Add the variables
            model_z_i = model.solver.interface.Variable(
                                        name=z_name,
                                        lb=0,
                                        ub=big_m,
                                        queue=False)
            if track_vars is not None:
                track_vars["vars"].append(model_z_i)
            model.solver.add([model_z_i])

            # z_i, cons = glovers_linearization(b = ga_i, fy=E, L=E.lb, U=E.ub, z=model_z_i)
            z_i, new_constraints = self.petersen_linearization(b=ga_i, x=E, M=big_m,
                                                        z=model_z_i)

            # Add the constraints:
            for cons in new_constraints:
                # Do not forget to substitute the sympy symbol in the constraint
                # with a variable  !
                # new_expression = cons.expression.subs(z_i, model_z_i.variable)
                # EDIT: Not anymore needed if we supply the variable

                new_cons = model.problem.Constraint(expression=cons["expression"], 
                                                    name=cons["name"],
                                                    ub=cons["ub"],
                                                    lb=cons["lb"])
                model.solver.add([new_cons])
                if track_vars is not None:
                    track_vars["cons"].append(new_cons)

            out_expr += (2 ** i) * mu_approx_resolution(mu_ub, mu_lb, n_mu_bins) * model_z_i
        out_expr += mu_lb * E

        cons = model.problem.Constraint(expression=out_expr, name='__GC__'.join(["sum_c", E.name]),
                            ub=0.,
                            lb=0.)

        model.solver.add([cons])
        if track_vars is not None:
            track_vars["cons"].append(cons)

        return out_expr

    def get_ordered_ga_vars(ga_vars):
        """
        Returns in order the variables that discretize growth
        :return:
        """
        # ga_i is a binary variable for the binary expansion f the fraction on N
        # of the max growth rate
        ga_vars = sorted(ga_vars, key=lambda x: int(x.name))
        return ga_vars

    def remove_vars_and_cons(self, to_remove):
        for con in to_remove["cons"]:
            if con in self.model.constraints:
                self.model.solver.remove(con)
        for var in to_remove["vars"]:
            if var in self.model.variables:
                self.model.solver.remove(var)
        
    
    def linearize_pycomo_model(self, f_vars, mu_lb=0., mu_ub=1000., n_mu_bins=10, track_vars=None):
        biomass_rxn = self.model.reactions.community_biomass
        member_names = self.get_member_names()
        ga_vars = self.init_mu_variables(self.model, biomass_rxn, mu_lb=mu_lb, mu_ub=mu_ub, n_mu_bins=n_mu_bins, track_vars=track_vars)
        for m in member_names: 
            self.linearize_me(
                model=self.model, 
                r_bio=self.model.reactions.get_by_id(f"{m}_to_community_biomass"), 
                f_var=f_vars[m], 
                ga_vars=ga_vars,
                mu_lb=mu_lb, 
                mu_ub=mu_ub, 
                n_mu_bins=n_mu_bins, 
                queue=False,
                track_vars=track_vars)
    
