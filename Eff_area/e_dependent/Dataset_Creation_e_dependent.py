import numpy as np
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel)
from MapDatasetNuisanceE_edependet import MapDatasetNuisanceE
from gammapy.modeling import Parameter, Parameters
path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'




class sys_dataset():
    def __init__(self, 
                 dataset_asimov,
                 tilt,
                 shift,
                 rnd):
        self.dataset_asimov = dataset_asimov
        self.shift = shift
        self.tilt = tilt
        self.rnd = rnd
        
    def set_model(self):
        models = Models.read(f"{path_crab}/standard_model.yml").copy()
        model_spectrum  = PowerLawSpectralModel(
            index=2.3,
            amplitude="1e-12 TeV-1 cm-2 s-1",    )
        source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                               spectral_model = model_spectrum,
                               name = "Source")  
        source_model.parameters['lon_0'].frozen = True
        source_model.parameters['lat_0'].frozen = True
        models = Models(source_model)
        return models
    
    def rel (self, x, t,s  ):
        return -t * x + (1+s) 
    
    def rel_3d (self, data, t, s ):
        rel_ = self.rel(x = np.arange(data.shape[0]), t = t, s =s)
        rel_3d = rel_[:,np.newaxis][:,np.newaxis]
        return data * rel_3d

    
    def create_dataset(self):
        dataset = self.dataset_asimov.copy()
        exposure = dataset.exposure.copy()
        exposure.data = self.rel_3d(data = exposure.data , t =  self.tilt, s = self.shift)
        background = dataset.background.copy()
        background.data = self.rel_3d(data = background.data , t =  self.tilt, s = self.shift)
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
        N_parameter1 = Parameter(name = "effareatilt", value = 0, frozen = False)
        N_parameter2 = Parameter(name = "effareashift", value = 0, frozen = False)
        
        N_parameters = Parameters([N_parameter1,N_parameter2])
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