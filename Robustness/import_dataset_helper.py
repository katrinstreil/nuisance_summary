
import yaml
import json
import numpy as np
import sys
from gammapy.maps import Map
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import Models, Model, FoVBackgroundModel
from gammapy.datasets import MapDataset #, MapDatasetNuisance
sys.path.append(
    "/home/hpc/caph/mppi045h/3D_analysis/"
)
from my_dataset_maps_19 import MapDatasetNuisance

outputfolder = 'output/data_asimov_tests'
path_local_repo_saturn = '/home/saturn/caph/mppi045h/Nuisance_Asimov_Datasets'
models = Models.read("1a-Source_models.yaml")


def read_mock_dataset(ad ):
    added = "_" + str(ad)
    dataset_N_sys = MapDatasetNuisance.read(f'{path_local_repo_saturn}/nui_dataset{added}.fits')
    with open(f'{path_local_repo_saturn}/nui_par{added}.yml', "r") as ymlfile:
        nui_par = yaml.load(ymlfile, Loader=yaml.FullLoader)
    dataset_N_sys.N_parameters = Parameters.from_dict(nui_par )
    bkg_model = FoVBackgroundModel(dataset_name=dataset_N_sys.name)
    models = Models([])
    models.append(bkg_model)
    dataset_N_sys.models =models
    return dataset_N_sys

class mocked_dataset():
    def __init__(self,spatial_model_type, amplitude, rnd, false_est):
        self.amplitude = amplitude
        self.rnd  = rnd
        self.spatial_model_type = spatial_model_type
        self.false_est = false_est
        self.dataset_N_sys = self.dataset_N_sys()
        
    def model_input(self):
        model_input = models[self.spatial_model_type].copy()
        model_input.parameters['amplitude'].value = self.amplitude.value
        return model_input  
    
    def sys_map(self):
        dataset_N_sys = self.dataset_N_sys.copy()
        sys_map = dataset_N_sys.N_map().copy()
        for e in range(24):
            ex = dataset_N_sys.exposure
            ex_ = ex.slice_by_idx(dict(energy_true= slice(e, e+1)))
            ex_.data = ex_.data / np.max(ex_.data)
            sys_map.slice_by_idx(dict(energy= slice(e, e+1))).data *= ex_.data
        return sys_map
    
    def npred_background(self):
        sys_map = self.sys_map()
        npred_background  = Map.from_geom(sys_map.geom)
        npred_background.data  = self.dataset_N_sys.background.data * (1+sys_map.data) 
        return npred_background
    
    def dataset_N_sys(self):
        dataset_N_sys = read_mock_dataset(self.rnd)
        dataset_N_sys.models = Models([self.model_input()])
        return dataset_N_sys

class fitted_datasets():
    def __init__(self,mocked_dataset):
        
        self.mocked_dataset = mocked_dataset
        self.dataset_N_sys = self.mocked_dataset.dataset_N_sys
        self.amplitude = self.mocked_dataset.amplitude
        self.rnd  = self.mocked_dataset.rnd
        self.spatial_model_type = self.mocked_dataset.spatial_model_type
        self.false_est = self.mocked_dataset.false_est
        self.outputfile = f'/OOutput{self.amplitude.value}.json'   
        
    def counts(self):
        npred = self.dataset_N_sys.npred_signal()
        npred.data += self.mocked_dataset.npred_background()
        return  npred
    
    def list_nuis(self):
        with open(f'{path_local_repo_saturn}/OOutput{self.amplitude.value}/nui_par_{self.rnd}.yml','r') as ymlfile:
            dict_nuis = yaml.load(ymlfile, Loader=yaml.FullLoader)
        list_nuis = []
        for case in dict_nuis.keys():
            list_nuis.append( Parameters.from_dict(dict_nuis[case] ))
        return list_nuis
    
    def result(self):
        with open(outputfolder+self.outputfile, 'r') as ymlfile:
             data = yaml.load(ymlfile, Loader=yaml.FullLoader)
        result = data[str(self.rnd)]['result']   
        return result
    
    def best_fit_models(self):
        model_input = self.dataset_N_sys.models[0].copy()
        best_fit_models = [model_input.copy(), model_input.copy()] 
        if self.false_est:
            best_fit_models = best_fit_models * 2
        result = self.result()
        par_names = model_input.parameters.names
        for i, par_name in enumerate(par_names):
            best_fit_models[0].parameters[par_name].value = result["best_fit_"+par_name+"_standard"] 
            best_fit_models[1].parameters[par_name].value = result["best_fit_"+par_name+"_N"] 
            best_fit_models[0].parameters[par_name].error =  result["best_fit_"+par_name+"_error_standard"] 
            best_fit_models[1].parameters[par_name].error =  result["best_fit_"+par_name+"_error_N"] 
            
            if self.false_est:
                best_fit_models[2].parameters[par_name].value = result["best_fit_"+par_name+"_N_under"] 
                best_fit_models[3].parameters[par_name].value = result["best_fit_"+par_name+"_N_over"] 
                best_fit_models[2].parameters[par_name].error =  result["best_fit_"+par_name+"_error_N_under"] 
                best_fit_models[3].parameters[par_name].error =  result["best_fit_"+par_name+"_error_N_over"] 
        return best_fit_models
        
    def datasets(self):
        datasets = [self.dataset_N_sys.copy(), self.dataset_N_sys.copy()]
        cases = ['standard','N']
        if self.false_est:
            datasets = datasets * 2
            cases = ['standard','N', "N_under", "N_over"]
        list_nuis = self.list_nuis()
        best_fit_models = self.best_fit_models()
        result = self.result()
        
        for i,d in enumerate(datasets):
            bkg_model = FoVBackgroundModel(dataset_name=d.name) 
            par_names = bkg_model.parameters.names
            case = cases[i]
            for par_name in par_names:
                bkg_model.parameters[par_name].value = result[f"best_fit_{par_name}_{case}"] 
                bkg_model.parameters[par_name].error = result[f"best_fit_{par_name}_error_{case}"] 
            
            models = Models([best_fit_models[i]])
            models.append(bkg_model)
            d.models = models
            d.counts = self.counts()
            if i >0: # do not set for standard dataset
                d.N_parameters = list_nuis[i-1]
            else:
                for n in d.N_parameters: # set all to zero:
                    n.value = 0
        return datasets
    
    
class analysis():
    def __init__(self, spatial_model_type, amplitude, rnds, false_est):
        self.spatial_model_type = spatial_model_type
        self.amplitude= amplitude
        self.rnds = rnds
        self.false_est = false_est
        self.get_datasets()
    def get_datasets(self):
        mocked_dataset_s = []
        datasets_standard = []
        datasets_corr =[]
        datasets_under = []
        datasets_over =[]
        for rnd in self.rnds:
            print(rnd)
            mocked_dataset_ = mocked_dataset('pointsource_center',
                                        self.amplitude, rnd ,self.false_est)
            mocked_dataset_s.append(mocked_dataset_)

            fitted_datasets_= fitted_datasets(mocked_dataset_ )
            datasets = fitted_datasets_.datasets()
            datasets_standard.append(datasets[0])
            datasets_corr.append(datasets[1])
            if self.false_est:
                datasets_under.append(datasets[2])
                datasets_over.append(datasets[3])
            else:
                datasets_under.append(None)
                datasets_over.append(None)
                
        self.mocked_dataset_s = mocked_dataset_s
        self.datasets_standard =datasets_standard
        self.datasets_corr =datasets_corr        
        self.datasets_under =datasets_under
        self.datasets_over =datasets_over