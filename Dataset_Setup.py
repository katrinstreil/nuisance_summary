import numpy as np
from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    PiecewiseNormSpectralModel,
    Models,
    PowerLawNormSpectralModel,
    MultiVariantePrior,
    GaussianPrior,
    CompoundNormSpectralModel,
    PowerLawNormOneHundredSpectralModel
)
from gammapy.modeling.models.IRF import (  
    EffAreaIRFModel,
    ERecoIRFModel,
    IRFModels
)
from gammapy.modeling import Parameters, Parameter
from scipy.stats import norm
from scipy.linalg import inv
import operator

class GaussianCovariance_matrix:
    def __init__(
        self,
        magnitude,  
        corrlength,
        size
        
    ):
        self.magnitude = magnitude
        self.corrlength = corrlength
        self.size = size
    
    def sys_percentage(self):
        return [self.magnitude for s in range(self.size)]
        
    def cov(self):
        zero = 1e-12
        cov = np.identity(self.size)
        sys_percentage = self.sys_percentage()
        # note: values set arbitrarily 
        for i in range(self.size):
            if sys_percentage[i] > 0:
                gau = norm.pdf(range(self.size) , loc = i , scale = self.corrlength )
                cov[i,:] = gau / np.max(gau) * sys_percentage[i] / 100 
                cov[i,:] += [zero] * (self.size)
        #carefull here it is hardcoded that the first 4 energybins are frozen
        idx = 4
        cov[:idx, :idx] = np.eye(idx)
        cov[idx:, :idx] = 0
        cov[:idx, idx:] = 0
        return cov
    
    def inv_cov(self):
        return inv(self.cov())
        
    def draw(self, seed):
        mean = np.zeros(self.size)
        # Generate the correlated values
        np.random.seed(seed)
        values = np.random.multivariate_normal(mean, self.cov(), size=1, )[0]
        # Scale the values to be between -0.1 and 0.1
        values = values / np.max(np.abs(values)) * self.magnitude / 100
        return values


class Setup:
    def __init__(
        self,
        dataset_input=None,
        rnd=False,
        e_reco_creation=10,
    ):
        self.dataset_input = dataset_input
        # set the sys parameters here and use npred as counts
        self.dataset_helper = self.set_up_dataset(name = "helper")
        self.dataset_helper.e_reco_n = e_reco_creation 
        self.rnd = rnd
        self.e_reco_creation = e_reco_creation
        self._irf_sys= False
        self._bkg_sys = False
        self._bkg_sys_V = False
        self._bkg_pl_sys_V = False
         
        
    def set_up_irf_sys(self, bias, resolution, norm, tilt): # onhunderd is the new eff arae model with e0 = 100 TeV and (1+ alpha)
        """
        Parameters:
        bias, resolution, norm, tilt
        
        sets irf_sys to True
        """
        self.bias = bias
        self.resolution = resolution
        self.norm= norm
        self.tilt =tilt
        self._irf_sys = True
    
        
    def set_up_bkg_sys(self, magnitude, corrlength, seed):
        """
        Parameters:
         magnitude [%], corrlength, seed
        
        sets _bkg_sys to True
        """
        self.magnitude = magnitude
        self.corrlength = corrlength
        self.seed= seed
        self._bkg_sys = True
        
    def set_up_bkg_sys_V(self, index1, index2, breake, magnitude):
        print("setup bkg V")
        self.index1 = index1
        self.index2 = index2
        self.breake = breake
        self.magnitude = magnitude
        self._bkg_sys_V = True
        
    def set_up_bkg_pl_sys_V(self, index1, index2, breake, magnitude):
        self.index1 = index1
        self.index2 = index2
        self.breake = breake
        self.magnitude = magnitude
        self._bkg_pl_sys_V = True
               
        
    def run(self):
        """
        Returns dataset and dataset_N
        both set up with the according models and filled with systematic
        """
        # set up datasets
        dataset, dataset_N = self.set_up_dataset(name = "dataset"), self.set_up_dataset(name = "dataset_N")
        # adding systematics if set before and setting irf/piecewise model to the dataset_N
        if self._irf_sys:
            self.add_irf_systematic(self.bias, self.resolution, self.norm, self.tilt)
            self.set_irf_model(dataset_N)
        if self._bkg_sys:
            # sets the counts
            self.add_bkg_systematic(self.magnitude, self.corrlength, self.seed)
            self.set_piecewise_bkg_model(dataset_N)
        elif self._bkg_sys_V:
            self.add_bkg_systematic_V( self.index1, self.index2, self.breake, self.magnitude)
            self.set_piecewise_bkg_model(dataset_N)
        elif self._bkg_pl_sys_V:
            self.add_bkg_systematic_V( self.index1, self.index2, self.breake, self.magnitude)
            self.set_piecewise_pl_bkg_model(dataset_N)
        else:            
            self.set_simple_bkg_model(dataset_N)
        
        self.set_simple_bkg_model(dataset)
        dataset.e_reco_n = self.e_reco_creation
        self.add_counts(dataset)
        if self.rnd:
            dataset_N.counts = dataset.counts
        else:
            self.add_counts(dataset_N)
        
        return dataset, dataset_N
        
    def set_up_dataset(self, name=None):
        """
        Returns dataset which is a copy of the input and the source model is set as model.
        """
        
        dataset = self.dataset_input.copy(name = name)
        models = Models(self.dataset_input.models.copy())
        dataset.models= models
        return dataset
    
    def set_simple_bkg_model(self, dataset):
        """
        sets the FOVbkgmodel to the rest of the models for the dataset
        """
        
        bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
        bkg_model.parameters["tilt"].frozen = False
        models = Models(dataset.models.copy())
        models.append(bkg_model)
        dataset.models = models
    
    def set_piecewise_bkg_model(self, dataset):
        """
        sets the FOVbkgmodel with the piece wise model as the spectral model to the rest of the models for the dataset
        """
       
        energy = dataset.geoms['geom'].axes[0].center
        l = len(energy)
        norms = Parameters([Parameter ("norm"+str(i), value = 0, frozen = False) for i in range(l)])
        
        piece = PiecewiseNormSpectralModel(energy = energy,
                                  norms = norms,
                                  interp="lin")
        piece.parameters['_norm'].value = 1
        bkg_model = FoVBackgroundModel(spectral_model = piece,
                                       dataset_name=dataset.name)
        models = Models(dataset.models.copy())
        models.append(bkg_model)
        dataset.models = models
        
        
        
    def set_piecewise_pl_bkg_model(self, dataset):
        """
        sets the FOVbkgmodel with the piece wise model as the spectral model to the rest of the models for the dataset
        """
       
        energy = dataset.geoms['geom'].axes[0].center
        l = len(energy)
        norms = Parameters([Parameter ("norm"+str(i), value = 0, frozen = False) for i in range(l)])
        piece = PiecewiseNormSpectralModel(energy = energy,
                                  norms = norms,
                                  interp="lin")
        compoundnorm = CompoundNormSpectralModel(
            model1=PowerLawNormSpectralModel(),
            model2=piece,
            operator=operator.add,
        )
        bkg_model = FoVBackgroundModel(spectral_model = compoundnorm,
                                       dataset_name=dataset.name)
        models = Models(dataset.models.copy())
        models.append(bkg_model)
        dataset.models = models
    
    def set_irf_model(self, dataset
                     ):
        """
        sets the IRF model to the rest of the models
        """
        # +1 in evaluation of PowerLawNormOneHundredSpectralModel
        # norm = 0 per default
        # E_0 = 100 TeV per default
        eff_area_model = EffAreaIRFModel(spectral_model = PowerLawNormOneHundredSpectralModel())
        # irf model
        IRFmodels = IRFModels(
            eff_area_model=eff_area_model, 
            e_reco_model=ERecoIRFModel(), 
            datasets_names=dataset.name
        )
        models = Models(dataset.models.copy())
        models.append(IRFmodels)
        dataset.models = models
         
    	
    def unset_model(self, dataset, modeltype):
        """
        unset the modeltype from all models attached to the dataset
        """
        models_set = Models(dataset.models.copy())
        models = Models()
        for m in models_set:
            if not isinstance(m, modeltype):
                models.append(m)
        dataset.models = models

    def add_irf_systematic(self, bias, resolution, norm, tilt):
        """
        sets IRF model , sets the model parameters as the input, sets the exposure and the edisp according to input
        removes the IRF model again
        """
       
        self.set_irf_model(self.dataset_helper)
        self.dataset_helper.irf_model.parameters['bias'].value = bias
        self.dataset_helper.irf_model.parameters['resolution'].value = resolution
        self.dataset_helper.irf_model.parameters['norm'].value = norm
        self.dataset_helper.irf_model.parameters['tilt'].value = tilt
        
        
    def emask(self):
        return self.dataset_helper.mask.data.sum(axis=2).sum(axis=1)>0
    
    def add_bkg_systematic(self, magnitude, corrlength, seed ):
        """
        sets piece wiese model, sets the model parameters as a draw from the cov. matrix
        computes the npred and sets as counts
        removes the piece wise model
        """
       
        Cov = GaussianCovariance_matrix(size = len(self.emask()),
                                magnitude = magnitude, 
                                corrlength = corrlength)
        cov  = Cov.cov()
        values = Cov.draw(seed)
        self.set_piecewise_bkg_model(self.dataset_helper)
        
        for n , v in zip(self.dataset_helper.background_model.parameters.free_parameters[self.emask()],
                         values[self.emask()]):
            n.value = v
            
      
    def add_bkg_systematic_V(self, index1, index2, breake, magnitude):
        print("add_bkg_systematic_V")
        self.set_piecewise_bkg_model(self.dataset_helper)
        N = len(self.dataset_helper.background_model.parameters.free_parameters[self.emask()])
        x_values = np.linspace(0, N-1,N)  
        values = [np.abs (x - breake)* index1 if x < breake else np.abs (x - breake)* index2 for x in x_values ]
        values /= np.max(values)
        values *= magnitude * 1e-2
        for n , v in zip(self.dataset_helper.background_model.parameters.free_parameters[self.emask()],
                         values):
            n.value = v

    
    
    def add_counts(self, dataset):
        """
        setting counts from the npred() with or without P. stat
        """
        npred = self.dataset_helper.npred()

        if self.rnd:
            if isinstance(self.rnd, int):
                print("set seed to:", self.rnd)
                np.random.seed(self.rnd)
            else:
                print("random seed")
                np.random.seed()
            counts_data = np.random.poisson(npred.data)
        else:
            counts_data = npred.data

        dataset.counts.data = counts_data
        self.dataset_helper.counts.data = counts_data
        
    def set_bkg_prior(self, dataset_asimov_N, magnitude, corrlength):
        """
        sets up multidim. prior for the piece wise bkg model
        """
        if isinstance(dataset_asimov_N.background_model.spectral_model, CompoundNormSpectralModel):
            modelparameters = dataset_asimov_N.background_model.spectral_model.model2.parameters
        else:
            modelparameters = dataset_asimov_N.background_model.parameters
        modelparameters = Parameters([m for m in modelparameters if m.name != "_norm" and m.name != "tilt" and m.name != "norm"])
        Cov = GaussianCovariance_matrix(size = len(self.emask()),
                                        magnitude = magnitude, 
                                        corrlength = corrlength)
        inv_cov  = Cov.inv_cov()
        multi_prior = MultiVariantePrior(modelparameters =modelparameters,
                             covariance_matrix = inv_cov,
                              name = "bkgsys"
                            )

    def set_irf_prior(self, dataset_asimov_N, bias, resolution, norm, tilt):
        """
        sets up Gaussian Priors for the IRF model parameters
        """
        simgas = {"bias":bias, "resolution":resolution, "norm":norm, "tilt":tilt}
        modelparameters = dataset_asimov_N.irf_model.parameters.free_parameters
        modelparameters = Parameters([m for m in modelparameters if m.name != "reference"])

        for m in modelparameters:
            GaussianPrior(modelparameters = m, mu = 0., sigma = simgas[m.name])



