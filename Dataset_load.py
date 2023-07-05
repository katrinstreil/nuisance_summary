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
source = 'Crab'

def create_asimov(cutoff = False, gun = False):
    models = Models.read(f"{path_crab}/standard_model.yml")
    dataset_asimov = MapDataset.read(f'{path}/{source}/stacked.fits')
    dataset_asimov = dataset_asimov.downsample(4)
    models = set_model(cutoff = cutoff, gun = gun)
    bkg_model = FoVBackgroundModel(dataset_name=dataset_asimov.name)
    bkg_model.parameters['tilt'].frozen  = False
    models.append(bkg_model)
    dataset_asimov.models = models
    dataset_asimov.counts = dataset_asimov.npred()
    return dataset_asimov

def set_model(cutoff = False, gun = False):
    models = Models.read(f"{path_crab}/standard_model.yml").copy()
    if cutoff:
        model_spectrum = ExpCutoffPowerLawSpectralModel(
            index=2.3,
            amplitude="1e-12 TeV-1 cm-2 s-1",
            lambda_ = "0.1 TeV-1")
    else:
        model_spectrum  = PowerLawSpectralModel(
            index=2.3,
            amplitude="1e-12 TeV-1 cm-2 s-1",    )
    if gun:
        gaus = GaussianSpectralModel(norm="1e-2 cm-2 s-1", 
                                 mean="2 TeV", 
                                 sigma="0.2 TeV")
        model_spectrum = CompoundSpectralModel(model_spectrum, gaus, operator.add)

    source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                           spectral_model = model_spectrum,
                           name = "Source")    
    source_model.parameters['lon_0'].frozen = True
    source_model.parameters['lat_0'].frozen = True
    models = Models(source_model)


    return models

def load_dataset_N(dataset_empty, path):
    models_load =  Models.read(path).copy()
    models = Models(models_load["Source"].copy())
    dataset_read = dataset_empty.copy()
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
                try:
                    p.value = models_load.parameters[p.name].value
                    p.error = models_load.parameters[p.name].error     
                except:
                    print(p.name, "not found")
            models.append(irf)
    models.append(bkg)
    dataset_read.models = models
    return dataset_read
