# %% [markdown]
# # PyCoMo Basics #
# PyCoMo is a **Py**thon **Co**mmunity metabolic **Mo**delling package. In this tutorial, the core features will be presented.
# 
# The expected runtime for this notebook is approximately 10-30 minutes.
# 
# ## Setting up PyCoMo ##
# Clone the package from github. Next, we are going to import all the packages we need in this tutorial.

# %%
from pathlib import Path
import sys
import cobra
import os

# %% [markdown]
# ### Importing PyCoMo ###

# %%
import pycomo
pycomo.configure_logger(level="info")

# %% [markdown]
# ## Creating a Community Model ##
# The creation of a community model consists of 3 steps:
# 1. Loading the member models
# 2. Preparing the member models for merging
# 3. Creating a community model
# 
# ### Loading the member models ###
# The community model creation process starts with models of the individual members. Note that the quality of the community model heavily depends on the quality of the member models!
# 
# In this tutorial we are using metabolic models from the AGORA collection. The models were retrieved from www.vmh.life, and are stored in the data folder of the repository. The selection of models and the resulting community represents a cystic fibrosis airway community, as done by Henson et al. (www.doi.org/10.1128/mSystems.00026-19)

# %%
test_model_dir = "../data/use_case/henson"
named_models = pycomo.load_named_models_from_dir(test_model_dir)

# %% [markdown]
# The models and file names were extracted and stored in named_models. Let's check the contents:

# %%
named_models

# %% [markdown]
# ### Preparing the models for merging ###
# With the models loaded, the next step is preparing them for merging. This is done by creating SingleOrganismModel objects. Using them, the models will be formatted for compliance with the SBML format. Further, an exchange compartment will be generated under the name _medium_.

# %% [markdown]
# One of the requirements for a community metabolic model is a common biomass function. To construct it, PyCoMo requires the biomass of each member represented as a single metabolite. This biomass metabolite ID can be specified when constructing the SingleOrganismModel objects. However, it can also be found or generated automatically, by setting the biomass reaction as the objective of the model. Let's check if the biomass function is the objective in all the models

# %%
for model in named_models.values():
    print(model.objective)

# %% [markdown]
# With the objective being the biomass function in all models, the biomass metabolite does not need to be specified.

# %%
single_org_models = []
for name, model in named_models.items():
    print(name)
    single_org_model = pycomo.SingleOrganismModel(model, name)
    single_org_models.append(single_org_model)

# %% [markdown]
# ### Creating a community model ###
# With the member models prepared, the community model can be generated. The first step is to create a CommunityModel objects from the member models. The matching of the exchange metabolites can be achieved in two ways: matching via identical metabolite IDs, or via annotation fields. In this tutorial and as all the models come from the same source, matching via identical metabolite IDs will be used.

# %%
community_name = "henson_community_model"
com_model_obj = pycomo.CommunityModel(single_org_models, community_name)

# %% [markdown]
# The cobra model of the community will generated the first time it is needed. We can enforce this now, by calling it via .model

# %%
com_model_obj.model

# %% [markdown]
# The output of the community model creation contains quite some lines of info and warnings. This is to be expected. Let's have a look at the different types of info:
# 1. _Ignoring reaction 'EX_4abz_medium' since it already exists._ This line will come up if a reaction is present in two different community member models under the same ID. This will only happen for exchange reactions in the exchange compartment and are therefor correct behaviour.
# 2. _WARNING: no annotation overlap found for matching metabolite mn2. Please make sure that the metabolite with this ID is indeed representing the same substance in all models!_ This warning comes up if exchange metabolites do not contain any matching annotation field. This can be an indicator that metabolites with the same ID are merged, but they represent different chemicals. Another common cause is that no annotation was given for this metabolite in one of the models.
# 3. _WARNING: matching of the metabolite CO2_EX is unbalanced (mass and/or charge). Please manually curate this metabolite for a mass and charge balanced model!_ This warning means that the formula of an exchange metabolite was different between member models. This can be due to the formula being omitted in some of the models. The other reason is that the metabolites differ in their mass or charge. As this would lead to generation or loss of matter from nothing, these issues need to be resolved for a consistent metabolic model.

# %% [markdown]
# ### Summary and report ###
# The community model object has two utility methods to display information on the model. 
# - Summary behaves the same as the summary method of COBRApy, displaying the the solution of FBA and its exchange metabolites. In the CommunityModel summary, the exchange reactions of metabolites responsible for scaling the flux bounds to the community composition are hidden.
# - The report function displays information on the model structure: the number of metabolites, reactions, genes, etc., but also quality control measures on mass and charge balance and internal loops.

# %%
com_model_obj.summary()

# %%
com_model_obj.report();

# %% [markdown]
# ### Setting the growth rate ###
# By default the community model object will have the structure of fixe growth rate. This means, the fractions of the community member abundance is allowed to vary during simulations, but the individual and community growth rate is set to a fixed value (default: 1.0). The next thing we will try is to set the community growth rate to a different value and do a FBA.

# %%
com_model_obj.apply_fixed_growth_rate(0.5)
com_model_obj.summary()

# %% [markdown]
# ### Setting the community member composition ###
# The model structure can be changed to fixed abundance, but variable growth rate. To do so, a conversion function needs to be called. Here we then change the community abundance to equal abundances.

# %%
com_model_obj.convert_to_fixed_abundance()
abundance_dict = com_model_obj.generate_equal_abundance_dict()
com_model_obj.apply_fixed_abundance(abundance_dict)
com_model_obj.summary()

# %% [markdown]
# ## Saving and loading community models ##
# Community model objects can be saved and loaded into SBML files. This is different from the other available option to save the cobra model of the community model objects, as the abundance fractions of the organisms are written into the file as well. Saving and loading the community model can be done like this:

# %%
com_model_obj.save("../data/toy/output/henson_com_model.xml")

# %%
com_model_obj_loaded = pycomo.CommunityModel.load("../data/toy/output/henson_com_model.xml")

# %%
com_model_obj_loaded

# %%
com_model_obj_loaded.model.optimize()

# %% [markdown]
# ### Quality Checks ###
# One of the quality checks that should be done is to look into all unbalanced reactions (mass and charge) in the entire model. As said before, such reactions should only exist in the case of boundary reactions, such as exchange, sink and source reactions.

# %%
unbalanced_reactions = com_model_obj.get_unbalanced_reactions()

print(f"{len(unbalanced_reactions)} unbalanced reactions")
print(f"Example reactions:")
for rxn in list(unbalanced_reactions.keys())[:10]:
    print(f"{rxn.id}: {unbalanced_reactions[rxn]}")

# %% [markdown]
# ## Analysis of community models ##
# PyCoMo offers the option to calculate all potential exchange metabolites and cross-feeding interactions in a community, independent of the community composition. The example for this part will be a three member community published by Koch et al. 2019 (https://doi.org/10.1371/journal.pcbi.1006759). The three member organisms are representatives of functional guilds in a biogas community.
# 
# ### Creating the community model ###
# We repeat the steps as before.

# %%
test_model_dir = "../data/use_case/koch"
named_models = pycomo.load_named_models_from_dir(test_model_dir)

# %%
named_models

# %%
single_org_models = []
for name, model in named_models.items():
    single_org_model = pycomo.SingleOrganismModel(model, name)
    single_org_models.append(single_org_model)
    
community_name = "koch_community_model"
com_model_obj = pycomo.CommunityModel(single_org_models, community_name)

# %% [markdown]
# With the community model generated, we set the medium for the analysis, as done by Koch et al.

# %%
medium = {
    'EX_CO2_EX_medium': 1000.0,
    'EX_Eth_EX_medium': 1000.0,
    'EX_BM_tot_medium': 1000.0
}
com_model_obj.medium = medium
com_model_obj.apply_medium()

# Some metabolites are not allowed to accumulate in the medium.
com_model_obj.model.reactions.get_by_id("EX_Form_EX_medium").upper_bound = 0.
com_model_obj.model.reactions.get_by_id("EX_H2_EX_medium").upper_bound = 0.

# %% [markdown]
# ### Calculating potential metabolite exchange ###
# All potential exchange metabolite fluxes and cross-feeding interactions can be calculated with the _potential_metabolite_exchanges_ method. This is a single FVA, but with a minimum objective of 0 and relaxed constraints. All reaction constraints are changed to include the value 0, which circumvents cases where a specific flux through a reaction is required, leading to infeasible solutions for certain community compositions.

# %%
com_model_obj.potential_metabolite_exchanges()

# %% [markdown]
# ### Plotting the maxiumum growth rate over the composition space ###

# %%
import pandas as pd

# Iterate over the fractions in steps of 0.01
com_model_obj.convert_to_fixed_abundance()
rows = []
for i in range (0,100,1):  # fraction of D. vulgaris
    for j in range (0, 100-i, 1): # fraction of M. hungatei
        if (100-i-j) < 0:
            continue

        abundances = {"dv": i/100., "mh": j/100., "mb": (100-i-j)/100.}
        
        # Apply the abuyndances
        com_model_obj.apply_fixed_abundance(abundances)
        
        # Reapply the bound restrictions of the exchange reactions
        com_model_obj.model.reactions.get_by_id("EX_Form_EX_medium").upper_bound = 0.
        com_model_obj.model.reactions.get_by_id("EX_H2_EX_medium").upper_bound = 0.
        
        # Calculate the optimal growth rate
        solution = com_model_obj.model.optimize()
        growth = 0. if str(solution.status) == "infeasible" else solution.objective_value
        rows.append({"dv": i/100., "mh": j/100., "growth": growth})
        
growth_df = pd.DataFrame(rows)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Restructure dataframe for heatmap
growth_df_pivot = growth_df.pivot(index="mh", columns="dv", values="growth")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(growth_df_pivot, ax=ax)
ax.invert_yaxis()

# %%



