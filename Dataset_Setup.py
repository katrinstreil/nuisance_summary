import numpy as np
from gammapy.datasets import MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    PiecewiseNormSpectralModel,
    Models,
    PowerLawNormSpectralModel,
    MultiVariantePrior,
    GaussianPrior
)
from gammapy.modeling.models.IRF import (  
    EffAreaIRFModel,
    ERecoIRFModel,
    IRFModels,
)
from gammapy.modeling import Parameters, Parameter
from scipy.stats import norm
from scipy.linalg import inv

class GaussianCovariance_matrix:
    def __init__(
        self,
        magnitude,  # in percent
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
        self.dataset_helper = self.set_up_dataset()

        self.rnd = rnd
        self.e_reco_creation = e_reco_creation
        self._irf_sys= False
        self._bkg_sys = False
        
         
        
    def set_up_irf_sys(self, bias, resolution, norm, tilt):
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
               
        
    def run(self):
        """
        Returns dataset and dataset_N
        both set up with the according models and filled with systematic
        """
        # set up datasets
        dataset, dataset_N = self.set_up_dataset(), self.set_up_dataset()
        # adding systematics if set before and setting irf/piecewise model to the dataset_N
        if self._irf_sys:
            self.add_irf_systematic(self.bias, self.resolution, self.norm, self.tilt)
            self.set_irf_model(dataset_N)
        if self._bkg_sys:
            # sets the counts
            self.add_bkg_systematic(self.magnitude, self.corrlength, self.seed)
            self.set_piecewise_bkg_model(dataset_N)
            
        else:            
            self.set_simple_bkg_model(dataset_N)
        
        self.set_simple_bkg_model(dataset)
        self.add_counts(dataset)
        if self.rnd:
            dataset_N.counts = dataset.counts
        else:
            self.add_counts(dataset_N)
        
        return dataset, dataset_N
        
    def set_up_dataset(self):
        """
        Returns dataset which is a copy of the input and the source model is set as model.
        """
        dataset = self.dataset_input.copy()
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
        bkg_model = FoVBackgroundModel(spectral_model = piece,
                                       dataset_name=dataset.name)
        models = Models(dataset.models.copy())
        models.append(bkg_model)
        dataset.models = models
    
    def set_irf_model(self, dataset):
        """
        sets the IRF model to the rest of the models
        """
       
        # irf model
        IRFmodels = IRFModels(
            eff_area_model=EffAreaIRFModel(spectral_model = PowerLawNormSpectralModel()), 
            e_reco_model=ERecoIRFModel(), 
            datasets_names=dataset.name
        )
        # +1 in evaluation 
        IRFmodels.parameters['norm'].value = 0.
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
        
    
    
    	
    def add_counts(self, dataset):
        """
        setting counts from the npred() with or without P. stat
        """
        npred = self.dataset_helper.npred()

        if self.rnd:
            counts_data = np.random.poisson(npred.data)
        else:
            counts_data = npred.data

        dataset.counts.data = counts_data
        
    def set_bkg_prior(self, dataset_asimov_N, magnitude, corrlength):
        """
        sets up multidim. prior for the piece wise bkg model
        """
        modelparameters = dataset_asimov_N.background_model.parameters
        modelparameters = Parameters([m for m in modelparameters if m.name != "_norm"])
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


        
