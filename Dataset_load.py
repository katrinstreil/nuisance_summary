import numpy as np
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel,
    ExpCutoffPowerLawSpectralModel
    )
from gammapy.modeling import Parameter, Parameters
from gammapy.datasets import MapDataset
from gammapy.modeling.models import SpectralModel
from gammapy.modeling.models.IRF import IRFModel, ERecoIRFModel, IRFModels, EffAreaIRFModel

import json
with open("/home/katrin/Documents/nuisance_summary/config.json") as json_data_file:
    config = json.load(json_data_file)

path = config['local']["path"]
path_crab = config['local']["path_crab"]
figformat = config['figformat']
source = 'Crab'

def create_asimov():
    models = Models.read(f"{path_crab}/standard_model.yml")
    dataset_asimov = MapDataset.read(f'{path}/{source}/stacked.fits')
    dataset_asimov = dataset_asimov.downsample(4)
    model_spectrum  = PowerLawSpectralModel(
        index=2.3,
        amplitude="1e-12 TeV-1 cm-2 s-1",    )
    source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                           spectral_model = model_spectrum,
                           name = "Source")    
    models = Models(source_model)
    bkg_model = FoVBackgroundModel(dataset_name=dataset_asimov.name)
    bkg_model.parameters['tilt'].frozen  = False
    models.append(bkg_model)
    dataset_asimov.models = models
    dataset_asimov.counts = dataset_asimov.npred()
    return dataset_asimov


def load_dataset_N(dataset_empty, path):
    models_load =  Models.read(path)
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
