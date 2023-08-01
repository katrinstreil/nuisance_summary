import numpy as np
import operator
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    GaussianSpectralModel,
    CompoundSpectralModel
    )
from gammapy.modeling.models import CompoundNormSpectralModel, PowerLawNormPenSpectralModel, PowerLawNormSpectralModel

from gammapy.modeling import Parameter, Parameters
from gammapy.datasets import MapDataset
from gammapy.modeling.models import SpectralModel
from gammapy.modeling.models.IRF import ERecoIRFModel, IRFModels, EffAreaIRFModel #,IRFModel
import Dataset_Creation
import json
import os 
path = os.getcwd()
substring = 'nuisance_summary'
path = path[:path.find(substring)] + substring +'/' 
with open(path+'config.json') as json_file:
    config = json.load(json_file)
case = config['case']
path = config[case]["path"]
path_crab = config[case]["path_crab"]
figformat = config['figformat']
path_pks = config[case]["path_pks"]


def create_asimov(model, source , parameters=None, livetime = None ):
    models = set_model(source, model)
      
    if livetime is not None:
        model = livetime
    if source == "Crab":
        dataset = MapDataset.read(f'{path_crab}/HESS_public/dataset-simulated-{model}.fits.gz')
    if source == "PKS":
        dataset = MapDataset.read(f'{path_pks}/HESS_public/dataset-simulated-{model}.fits.gz')
        
    if parameters is not None:
        for p in parameters:
            models.parameters[p.name].value = p.value
            models.parameters[p.name].error = p.error
            
    bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
    bkg_model.parameters['tilt'].frozen  = False
    models.append(bkg_model)
    dataset.models = models
    dataset.counts = dataset.npred()
    return dataset
    

def set_model(source, model):
    if source == "Crab":
        return Models.read(f'{path_crab}/HESS_public/model-{model}.yaml').copy()
    if source == "PKS":
        return Models.read(f'{path_pks}/HESS_public/model-{model}.yaml').copy()

def load_dataset_N(dataset_empty, path, bkg_sys = False):
    models_load =  Models.read(path).copy()
    Source = models_load.names[0]
    models = Models(models_load[Source].copy())
    dataset_read = dataset_empty.copy()
    
    if bkg_sys:
        import operator
        compoundnorm  = CompoundNormSpectralModel(model1  = PowerLawNormSpectralModel(),
                                                 model2 = PowerLawNormPenSpectralModel(),
                                                 operator =  operator.mul)

        bkg = FoVBackgroundModel(dataset_name=dataset_read.name,
                                spectral_model = compoundnorm)
      
    else:
        bkg = FoVBackgroundModel( dataset_name = dataset_read.name)
    for p in bkg.parameters:
        p.value = models_load.parameters[p.name].value
        p.error = models_load.parameters[p.name].error    
    for m in models_load:
        if m.type=='irf':
            irf = IRFModels(e_reco_model = ERecoIRFModel(),
                            eff_area_model = EffAreaIRFModel(),
                           datasets_names = dataset_read.name)
            for p in irf.parameters:
                p.frozen = False
                try:
                    p.value = models_load.parameters[p.name].value
                    p.error = models_load.parameters[p.name].error     
                    p.frozen = models_load.parameters[p.name].frozen     
                    
                except:
                    print(p.name, "not found")
            models.append(irf)
    models.append(bkg)
    dataset_read.models = models
    return dataset_read


def load_dataset(dataset_empty, path):
    models_load =  Models.read(path).copy()
    Source = models_load.names[0]
    models = Models(models_load[Source].copy())
    dataset_read = dataset_empty.copy()
    
    bkg = FoVBackgroundModel( dataset_name = dataset_read.name)
    for p in bkg.parameters:
        p.value = models_load.parameters[p.name].value
        p.error = models_load.parameters[p.name].error    

    models.append(bkg)
    dataset_read.models = models
    return dataset_read