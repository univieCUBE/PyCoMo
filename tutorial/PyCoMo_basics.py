#!/usr/bin/env python
# coding: utf-8

# # PyCoMo Basics #
# PyCoMo is a **Py**thon **Co**mmunity metabolic **Mo**delling package. In this tutorial, the core features will be presented.
# 
# The expected runtime for this notebook is approximately 5 minutes.
# ## Setting up PyCoMo ##
# Clone the package from github. Next, we are going to import all the packages we need in this tutorial.

# In[1]:


from pathlib import Path
import sys
import cobra
import os


# ### Importing PyCoMo ###
# As PyCoMo is currently only available as a local package, the direct path to the package directory needs to be used on import.

# In[2]:


path_root = "../pycomo"  # Change path according to your PyCoMo location
sys.path.append(str(path_root))
import pycomo as pycomo


# Now we will check if PyCoMo was loaded correctly. For this, we will run the help function on the PyCoMo package.

# In[ ]:


help(pycomo)


# ## Creating a Community Model ##
# The creation of a community model consists of 3 steps:
# 1. Loading the member models
# 2. Preparing the member models for merging
# 3. Creating a community model
# ### Loading the member models ###
# The community model creation process starts with models of the individual members. Note that the quality of the community model heavily depends on the quality of the member models!
# 
# In this tutorial we are using metabolic models from the AGORA collection. The models were retrieved from www.vmh.life, and are stored in the data folder of the repository. The selection of models and the resulting community represents a cystic fibrosis airway community, as done by Henson et al. (www.doi.org/10.1128/mSystems.00026-19)

# In[4]:


test_model_dir = "../data/use_case/henson"
named_models = pycomo.load_named_models_from_dir(test_model_dir)


# The models and file names were extracted and stored in named_models. Let's check the contents:

# In[5]:


named_models


# ### Preparing the models for merging ###
# With the models loaded, the next step is preparing them for merging. This is done by creating SingleOrganismModel objects. Using them, the models will be formatted for compliance with the SBML format. Further, an exchange compartment will be generated under the name _exchg_.

# One of the requirements for a community metabolic model is a common biomass function. To construct it, PyCoMo requires the biomass of each member represented as a single metabolite. This biomass metabolite ID can be specified when constructing the SingleOrganismModel objects. However, it can also be found or generated automatically, by setting the biomass reaction as the objective of the model. Let's check if the biomass function is the objective in all the models

# In[6]:


for model in named_models.values():
    print(model.objective)


# With the objective being the biomass function in all models, the biomass metabolite does not need to be specified.

# In[7]:


single_org_models = []
for name, model in named_models.items():
    print(name)
    single_org_model = pycomo.SingleOrganismModel(model, name)
    single_org_models.append(single_org_model)


# ### Creating a community model ###
# With the member models prepared, the community model can be generated. The first step is to create a CommunityModel objects from the member models. The matching of the exchange metabolites can be achieved in two ways: matching via identical metabolite IDs, or via annotation fields. In this tutorial and as all the models come from the same source, matching via identical metabolite IDs will be used.

# In[8]:


community_name = "henson_community_model"
com_model_obj = pycomo.CommunityModel(single_org_models, community_name)


# The cobra model of the community will generated the first time it is needed. We can enforce this now, by calling it via .community_model

# In[9]:


com_model_obj.community_model


# The output of the community model creation contains quite some lines of info and warnings. This is to be expected. Let's have a look at the different types of info:
# 1. _Ignoring reaction 'EX_4abz_exchg' since it already exists._ This line will come up if a reaction is present in two different community member models under the same ID. This will only happen for exchange reactions in the exchange compartment and are therefor correct behaviour.
# 2. _WARNING: no annotation overlap found for matching metabolite mn2. Please make sure that the metabolite with this ID is indeed representing the same substance in all models!_ This warning comes up if exchange metabolites do not contain any matching annotation field. This can be an indicator that metabolites with the same ID are merged, but they represent different chemicals. Another common cause is that no annotation was given for this metabolite in one of the models.
# 3. _WARNING: matching of the metabolite CO2_EX is unbalanced (mass and/or charge). Please manually curate this metabolite for a mass and charge balanced model!_ This warning means that the formula of an exchange metabolite was different between member models. This can be due to the formula being omitted in some of the models. The other reason is that the metabolites differ in their mass or charge. As this would lead to generation or loss of matter from nothing, these issues need to be resolved for a consistent metabolic model.

# ### Setting the community member composition ###
# For the bounds of the model and the normalisation to be correct, the fractions of all community members must be set (and sum up to 1.0). A quick way to do this is to set the abundance fractions equal for all community members.

# In[10]:


com_model_obj.equal_abundance()


# Now let us check if the biomass function was updated accordingly as well

# In[11]:


com_model_obj.community_model.reactions.get_by_id("community_biomass").reaction


# As can be seen above, the biomass function now takes an equal amount of all 17 community members, 1/17th or 0.0588...

# ### Quality Checks ###
# One of the quality checks that should be done is to look into all unbalanced reactions (mass and charge) in the entire model. As said before, such reactions should only exist in the case of boundary reactions, such as exchange, sink and source reactions.

# In[12]:


com_model_obj.get_unbalanced_reactions()


# ## Saving and loading community models ##
# Community model objects can be saved and loaded into SBML files. This is different from the other available option to save the cobra model of the community model objects, as the abundance fractions of the organisms are written into the file as well. Saving and loading the community model can be done like this:

# In[13]:


com_model_obj.save("../data/toy/output/henson_com_model.xml")


# In[14]:


com_model_obj_loaded = pycomo.CommunityModel.load("../data/toy/output/henson_com_model.xml")


# In[15]:


com_model_obj_loaded.community_model.summary()


# ## Analysis of community models ##
# Work in progress.
