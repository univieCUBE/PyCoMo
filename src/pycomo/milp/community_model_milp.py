from pycomo.pycomo_models import CommunityModel

class CommunityModelMILP(CommunityModel):
    """
    Community model class with additional functionalities, which depend on MILP solvers.
    Functionalities include: Binarized growth rate, exclusion of thermodynamically infeasible loops.
    """