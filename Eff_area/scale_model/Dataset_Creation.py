import numpy as np
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel,
    PowerLawNuisanceSpectralModel,
    PowerLawNormNuisanceSpectralModel)
#from MapDatasetNuisanceE import MapDatasetNuisanceE
from gammapy.modeling import Parameter, Parameters
from gammapy.datasets import MapDataset

path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'

from gammapy.modeling.models import SpectralModel

class NuisanceNorm(SpectralModel):

    def __init__(self, model):
        self.eff_area = Parameter("eff_area", value =  0.,  is_penalised = True)
        self.model = model
        self.norm =[p for p in model.parameters if p.is_norm]
    
    @property    
    def parameters(self):
        return  Parameters.from_stack([Parameters([self.eff_area]),  self.model.parameters])
        
    # maybe self here is wrong:?    
    #@staticmethod
    def evaluate(self, energy, **kwargs):
        eff_area_ = kwargs.pop('eff_area')
        # todo extract is_norm parameter name
        for norm_parameter in self.norm:
            kwargs[norm_parameter.name] *= (1.+eff_area_)
        return self.model.evaluate(energy, **kwargs) 
    
    

class sys_dataset():
    def __init__(self, 
                 dataset_asimov,
                 shift,
                 tilt,
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
        #background.data = self.rel_3d(data = background.data , t =  self.tilt, s = self.shift)
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
    
    
    def set_model_N(self):
        models = Models.read(f"{path_crab}/standard_model.yml").copy()
        powerlaw = PowerLawSpectralModel(index=2.3,
                                         amplitude="1e-12 TeV-1 cm-2 s-1",  
                                         name = "Source")
        norm_nuisance = NuisanceNorm(model = powerlaw)
        from gammapy.modeling import Fit,  Parameters, Covariance , Parameter

        norm_nuisance._covariance = Covariance(norm_nuisance.parameters)
        
        source_model = SkyModel(spatial_model = models['main source'].spatial_model ,
                               spectral_model = norm_nuisance,
                               name = "SourceN")  
        source_model.parameters['lon_0'].frozen = True
        source_model.parameters['lat_0'].frozen = True
        models = Models(source_model)
        return models



    def create_dataset_N(self):
        dataset_ = self.create_dataset()
        dataset_N = MapDataset(
                counts=dataset_.counts.copy(),
                exposure=dataset_.exposure.copy(),
                background=dataset_.background.copy(),
                psf=dataset_.psf.copy(),
                edisp=dataset_.edisp.copy(),
                mask_safe=dataset_.mask_safe.copy(),
                gti=dataset_.gti.copy(),
                name='dataset N')
        models = self.set_model_N()
        bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name,
                                      )#spectral_model = bkg_spectralmodel)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset_N.models = models
        return dataset_N