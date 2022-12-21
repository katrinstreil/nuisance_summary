import numpy as np
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel)
from MapDatasetNuisanceE import MapDatasetNuisanceE
from gammapy.modeling import Parameter, Parameters
path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'




class sys_dataset():
    def __init__(self, 
                 dataset_asimov,
                 factor,
                 rnd):
        self.dataset_asimov = dataset_asimov
        self.factor = factor
        self.rnd = rnd
        
    def set_model(self):
        models = Models.read(f"{path_crab}/standard_model.yml").copy()
        model_spectrum  = PowerLawSpectralModel(
            index=2.3,
            amplitude="1e-12 TeV-1 cm-2 s-1",    )
        source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                               spectral_model = model_spectrum,
                               name = "Source")    
        models = Models(source_model)
        return models
    
    def create_dataset(self):
        dataset = self.dataset_asimov.copy()
        exposure = dataset.exposure.copy()
        exposure.data *= (1-self.factor)
        background = dataset.background.copy()
        background.data *= (1-self.factor)
        dataset.exposure = exposure
        dataset.background = background
        if self.rnd:
            counts_data = np.random.poisson(self.dataset_asimov.counts.data)
        else:
            counts_data = self.dataset_asimov.counts.data

        dataset.counts.data = counts_data
        models = self.set_model()
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset.models = models
        return dataset
    
    
    def create_dataset_N(self, sigma):
        dataset_ = self.create_dataset()
        N_parameter = Parameter(name = "effarea", value = 0)
        N_parameters = Parameters([N_parameter])
        dataset_N = MapDatasetNuisanceE (
                counts=dataset_.counts.copy(),
                exposure=dataset_.exposure.copy(),
                background=dataset_.background.copy(),
                psf=dataset_.psf.copy(),
                edisp=dataset_.edisp.copy(),
                mask_safe=dataset_.mask_safe.copy(),
                gti=dataset_.gti.copy(),
                name='dataset N',
                N_parameters=N_parameters,
                penalty_sigma= sigma)
        models = self.set_model()
        bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset_N.models = models
        return dataset_N