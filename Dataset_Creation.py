import numpy as np
import operator
from gammapy.modeling.models import (
    Models, 
    FoVBackgroundModel,
    PowerLawSpectralModel,
    SkyModel,
    ExpCutoffPowerLawSpectralModel,
    GaussianSpectralModel,
    CompoundSpectralModel,
    CompoundNormSpectralModel,
    PowerLawNormSpectralModel,
    PowerLawNormPenSpectralModel
    )
from gammapy.modeling import Parameter, Parameters
from gammapy.datasets import MapDataset

#path_crab = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/nuisance_summary/Crab'
path_crab = '/home/katrin/Documents/Crab'
from gammapy.modeling.models import SpectralModel
from gammapy.modeling.models.IRF import ERecoIRFModel, IRFModels, EffAreaIRFModel #,IRFModel

class sys_dataset():
    def __init__(self, 
                 dataset_asimov=None,
                 shift=0,
                 tilt=0,
                 bias=0,
                 resolution=0,
                 bkg_norm=None, 
                 bkg_tilt=None,
                 rnd=False,
                 e_reco_creation=10,
                 cutoff = False,
                 gun = False):
        self.dataset_asimov = dataset_asimov
        self.shift = shift
        self.tilt = tilt
        self.bias = bias
        self.resolution = resolution
        self.bkg_norm = bkg_norm
        self.bkg_tilt = bkg_tilt
        self.rnd = rnd
        self.e_reco_creation = e_reco_creation
        self.cutoff = cutoff
        self.gun = gun
        
    def create_dataset(self):
        dataset = self.dataset_asimov.copy()
        models = Models(self.dataset_asimov.models.copy())
        #bkg model
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        dataset.models = models
        
        if self.rnd:
            counts_data = np.random.poisson(dataset.npred().data)
        else:
            counts_data = dataset.npred().data

        dataset.counts.data = counts_data
        
        #irf model
        dataset.e_reco_n = self.e_reco_creation 
        effareamodel = EffAreaIRFModel()
        ereco = ERecoIRFModel()
        IRFmodels = IRFModels(eff_area_model= effareamodel,
                              e_reco_model= ereco,
                             datasets_names = dataset.name)
        models.append(IRFmodels)
        dataset.models = models
        dataset.models.parameters['norm_nuisance'].value  = self.shift
        dataset.models.parameters['tilt_nuisance'].value  = self.tilt
        dataset.models.parameters['bias'].value  = self.bias
        dataset.models.parameters['resolution'].value  = self.resolution
        dataset.exposure = dataset.npred_exposure()
        dataset.edisp = dataset.npred_edisp()

        # set models without the IRF model
        models = Models(self.dataset_asimov.models.copy())
        models.append(FoVBackgroundModel(dataset_name=dataset.name))
        models.parameters['tilt'].frozen = False
        dataset.models = models
        
        return dataset
    
    

    def create_dataset_N(self, e_reco_n):
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
        models = Models(self.dataset_asimov.models.copy())
        #bkg model
        if self.bkg_norm is not None or self.bkg_tilt is not None:
            import operator
            model2 = PowerLawNormPenSpectralModel()
            compoundnorm  = CompoundNormSpectralModel(model1  = PowerLawNormSpectralModel(),
                                                     model2 = model2,
                                                     operator =  operator.mul)
            
            bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name,
                                    spectral_model = compoundnorm)
            if self.bkg_norm is not None: 
                bkg_model.parameters['norm_nuisance'].value  = self.bkg_norm
            if self.bkg_tilt is not None:
                bkg_model.parameters['tilt_nuisance'].value  = self.bkg_tilt

        else:
            bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name)
            
        bkg_model.parameters['tilt'].frozen  = False
        models.append(bkg_model)
        #irf model
        effareamodel = EffAreaIRFModel()
        ereco = ERecoIRFModel()
        IRFmodels = IRFModels(eff_area_model= effareamodel,
                              e_reco_model= ereco,
                             datasets_names = dataset_N.name)
 
        models.append(IRFmodels)
        dataset_N.models = models
        dataset_N.e_reco_n = e_reco_n
        return dataset_N