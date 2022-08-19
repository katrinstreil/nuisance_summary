
import yaml
import json
import numpy as np
outputfolder = 'output/data_asimov_tests'

class analysis():
    def __init__(self, spatial_model_type, amplitude, rnds, false_est):
        self.spatial_model_type = spatial_model_type
        outputfile = 'wrongmodeltype'
        if "pointsource_center" in self.spatial_model_type:
            outputfile = f'/OOutput{amplitude}.json'   
            self.title = "Point Source (Center)"
        self.amplitude  = amplitude
        self.input_norm = 1
        self.outputfolder = outputfolder +outputfile
        with open(self.outputfolder, 'r') as f:
            self.data = json.load(f)
        self.rnds = [(a) for a in self.data.keys()]
        self.amplitudes_x = rnds #[(a) for a in self.data.keys()]
        self.rnds = rnds
        self.false_est = false_est
    def significance (self):
        self.TS_standard  =[]
        self.TS_N_under = []
        self.TS_N  =[]
        self.TS_N_over  =[]
        

        for a in self.rnds:
            self.TS_standard.append(self.data[str(a)]['result']['TS_standard'])
            self.TS_N.append(self.data[str(a)]['result']['TS_N'])
            if self.false_est:
                self.TS_N_under.append(self.data[str(a)]['result']['TS_N_under'])
                self.TS_N_over.append(self.data[str(a)]['result']['TS_N_over'])
            else:
                self.TS_N_under.append(np.nan)
                self.TS_N_over.append(np.nan)
                
        self.mean_TS_standard = np.nanmean(self.TS_standard)
        self.mean_TS_N_under = np.nanmean(self.TS_N_under)
        self.mean_TS_N = np.nanmean(self.TS_N)
        self.mean_TS_N_over = np.nanmean(self.TS_N_over)
        self.std_TS_standard = np.nanstd(self.TS_standard)
        self.std_TS_N_under = np.nanstd(self.TS_N_under)
        self.std_TS_N = np.nanstd(self.TS_N)
        self.std_TS_N_over = np.nanstd(self.TS_N_over)
        
        
    def best_fit_amplitudes (self):
        
        self.x_mins = []
        self.x_maxs = []        
        self.err_standard = []
        self.ampli_standard = []

        self.x_mins_N = []
        self.x_maxs_N = []  
        self.err_N = []
        self.ampli_N = []
        
        self.x_mins_N_under = []
        self.x_maxs_N_under = []  
        self.err_N_under = []
        self.ampli_N_under = []
        
        self.x_mins_N_over = []
        self.x_maxs_N_over = []  
        self.err_N_over = []
        self.ampli_N_over = []

        for a in self.rnds:
            a = str(a)
            err = self.data[(a)]['result']['best_fit_amplitude_error_standard']
            ampli = self.data[(a)]['result']['best_fit_amplitude_standard']
            x_min = ampli/float(self.amplitude) - err /float(self.amplitude)
            x_max = ampli/float(self.amplitude) + err /float(self.amplitude)
            self.x_mins.append(x_min)
            self.x_maxs.append(x_max)
            self.err_standard.append(err/float(self.amplitude))
            self.ampli_standard.append(ampli/float(self.amplitude))
            
            err = self.data[(a)]['result']['best_fit_amplitude_error_N']
            ampli = self.data[(a)]['result']['best_fit_amplitude_N']
            x_min_N = ampli/float(self.amplitude) - err /float(self.amplitude)
            x_max_N = ampli/float(self.amplitude) + err /float(self.amplitude)
            self.x_mins_N.append(x_min_N)
            self.x_maxs_N.append(x_max_N)
            self.err_N.append(err/float(self.amplitude))
            self.ampli_N.append(ampli/float(self.amplitude))
            
        
            
            if self.false_est:
            
                err = self.data[(a)]['result']['best_fit_amplitude_error_N_under']
                ampli = self.data[(a)]['result']['best_fit_amplitude_N_under']
                x_min_N_under = ampli/float(self.amplitude) - err /float(self.amplitude)
                x_max_N_under = ampli/float(self.amplitude) + err /float(self.amplitude)
                self.x_mins_N_under.append(x_min_N_under)
                self.x_maxs_N_under.append(x_max_N_under)
                self.err_N_under.append(err/float(self.amplitude))
                self.ampli_N_under.append(ampli/float(self.amplitude))

                err = self.data[(a)]['result']['best_fit_amplitude_error_N_over']
                ampli = self.data[(a)]['result']['best_fit_amplitude_N_over']
                x_min_N_over = ampli/float(self.amplitude) - err /float(self.amplitude)
                x_max_N_over = ampli/float(self.amplitude) + err /float(self.amplitude)
                self.x_mins_N_over.append(x_min_N_over)
                self.x_maxs_N_over.append(x_max_N_over)
                self.err_N_over.append(err/float(self.amplitude))
                self.ampli_N_over.append(ampli/float(self.amplitude))
                
            else:
                self.x_mins_N_under.append(np.nan)
                self.x_maxs_N_under.append(np.nan)
                self.err_N_under.append(np.nan)
                self.ampli_N_under.append(np.nan)
                self.x_mins_N_over.append(np.nan)
                self.x_maxs_N_over.append(np.nan)
                self.err_N_over.append(np.nan)
                self.ampli_N_over.append(np.nan)
                
        self.mean_x_mins = np.nanmean(self.x_mins)
        self.mean_x_maxs = np.nanmean(self.x_maxs)        
        self.mean_err_standard = np.nanmean(self.err_standard)
        self.mean_ampli_standard = np.nanmean(self.ampli_standard)
        self.std_x_mins = np.nanstd(self.x_mins)
        self.std_x_maxs = np.nanstd(self.x_maxs)        
        self.std_err_standard = np.nanstd(self.err_standard)
        self.std_ampli_standard = np.nanstd(self.ampli_standard)

        self.mean_x_mins_N = np.nanmean(self.x_mins_N)
        self.mean_x_maxs_N = np.nanmean(self.x_maxs_N)  
        self.mean_err_N = np.nanmean(self.err_N)
        self.mean_ampli_N = np.nanmean(self.ampli_N)
        self.std_x_mins_N = np.nanstd(self.x_mins_N)
        self.std_x_maxs_N = np.nanstd(self.x_maxs_N)  
        self.std_err_N = np.nanstd(self.err_N)
        self.std_ampli_N = np.nanstd(self.ampli_N)

        self.mean_x_mins_N_under = np.nanmean(self.x_mins_N_under)
        self.mean_x_maxs_N_under = np.nanmean(self.x_maxs_N_under)  
        self.mean_err_N_under = np.nanmean(self.err_N_under)
        self.mean_ampli_N_under = np.nanmean(self.ampli_N_under)
        self.std_x_mins_N_under = np.nanstd(self.x_mins_N_under)
        self.std_x_maxs_N_under = np.nanstd(self.x_maxs_N_under)  
        self.std_err_N_under = np.nanstd(self.err_N_under)
        self.std_ampli_N_under = np.nanstd(self.ampli_N_under)
    
        self.mean_x_mins_N_over = np.nanmean(self.x_mins_N_over)
        self.mean_x_maxs_N_over = np.nanmean(self.x_maxs_N_over)  
        self.mean_err_N_over = np.nanmean(self.err_N_over)
        self.mean_ampli_N_over = np.nanmean(self.ampli_N_over)
        self.std_x_mins_N_over = np.nanstd(self.x_mins_N_over)
        self.std_x_maxs_N_over = np.nanstd(self.x_maxs_N_over)  
        self.std_err_N_over = np.nanstd(self.err_N_over)
        self.std_ampli_N_over = np.nanstd(self.ampli_N_over)
                
    def best_fit_norm (self):
        self.norm_x_mins = []
        self.norm_x_maxs = []        
        self.norm_err_standard = []
        self.norm_standard = []

        self.norm_x_mins_N = []
        self.norm_x_maxs_N = []  
        self.norm_err_N = []
        self.norm_N = []
        
        self.norm_x_mins_N_under = []
        self.norm_x_maxs_N_under = []  
        self.norm_err_N_under = []
        self.norm_N_under = []
        
        self.norm_x_mins_N_over = []
        self.norm_x_maxs_N_over = []  
        self.norm_err_N_over = []
        self.norm_N_over = []

        for a in self.rnds:
            a = str(a)
            err = self.data[(a)]['result']['best_fit_norm_error_standard']
            norm = self.data[(a)]['result']['best_fit_norm_standard']
            x_min = norm/float(self.input_norm) - err /float(self.input_norm)
            x_max = norm/float(self.input_norm) + err /float(self.input_norm)
            self.norm_x_mins.append(x_min)
            self.norm_x_maxs.append(x_max)
            self.norm_err_standard.append(err/float(self.input_norm))
            self.norm_standard.append(norm/float(self.input_norm))

            err = self.data[(a)]['result']['best_fit_norm_error_N']
            norm = self.data[(a)]['result']['best_fit_norm_N']
            x_min_N = norm/float(self.input_norm) - err /float(self.input_norm)
            x_max_N = norm/float(self.input_norm) + err /float(self.input_norm)
            self.norm_x_mins_N.append(x_min_N)
            self.norm_x_maxs_N.append(x_max_N)
            self.norm_err_N.append(err/float(self.input_norm))
            self.norm_N.append(norm/float(self.input_norm))
            
            if self.false_est:
            
                err = self.data[(a)]['result']['best_fit_norm_error_N_under']
                norm = self.data[(a)]['result']['best_fit_norm_N_under']
                x_min_N_under = norm/float(self.input_norm) - err /float(self.input_norm)
                x_max_N_under = norm/float(self.input_norm) + err /float(self.input_norm)
                self.norm_x_mins_N_under.append(x_min_N_under)
                self.norm_x_maxs_N_under.append(x_max_N_under)
                self.norm_err_N_under.append(err/float(self.input_norm))
                self.norm_N_under.append(norm/float(self.input_norm))

                err = self.data[(a)]['result']['best_fit_norm_error_N_over']
                norm = self.data[(a)]['result']['best_fit_norm_N_over']
                x_min_N_over = norm/float(self.input_norm) - err /float(self.input_norm)
                x_max_N_over = norm/float(self.input_norm) + err /float(self.input_norm)
                self.norm_x_mins_N_over.append(x_min_N)
                self.norm_x_maxs_N_over.append(x_max_N)
                self.norm_err_N_over.append(err/float(self.input_norm))
                self.norm_N_over.append(norm/float(self.input_norm))
            else:
                self.norm_x_mins_N_under.append(np.nan)
                self.norm_x_maxs_N_under.append(np.nan)
                self.norm_err_N_under.append(np.nan)
                self.norm_N_under.append(np.nan)
                self.norm_x_mins_N_over.append(np.nan)
                self.norm_x_maxs_N_over.append(np.nan)
                self.norm_err_N_over.append(np.nan)
                self.norm_N_over.append(np.nan)

        self.mean_norm_x_mins = np.nanmean(self.norm_x_mins)
        self.mean_norm_x_maxs = np.nanmean(self.norm_x_maxs)        
        self.mean_norm_err_standard = np.nanmean(self.norm_err_standard)
        self.mean_norm_standard = np.nanmean(self.norm_standard)

        self.std_norm_x_mins = np.nanstd(self.norm_x_mins)
        self.std_norm_x_maxs = np.nanstd(self.norm_x_maxs)        
        self.std_norm_err_standard = np.nanstd(self.norm_err_standard)
        self.std_norm_standard = np.nanstd(self.norm_standard)

        self.mean_norm_x_mins_N = np.nanmean(self.norm_x_mins_N)
        self.mean_norm_x_maxs_N = np.nanmean(self.norm_x_maxs_N)  
        self.mean_norm_err_N = np.nanmean(self.norm_err_N)
        self.mean_norm_N = np.nanmean(self.norm_N)

        self.std_norm_x_mins_N = np.nanstd(self.norm_x_mins_N)
        self.std_norm_x_maxs_N = np.nanstd(self.norm_x_maxs_N)  
        self.std_norm_err_N = np.nanstd(self.norm_err_N)
        self.std_norm_N = np.nanstd(self.norm_N)


        self.mean_norm_x_mins_N_under= np.nanmean(self.norm_x_mins_N_under)
        self.mean_norm_x_maxs_N_under= np.nanmean(self.norm_x_maxs_N_under)  
        self.mean_norm_err_N_under= np.nanmean(self.norm_err_N_under)
        self.mean_norm_N_under= np.nanmean(self.norm_N_under)

        self.std_norm_x_mins_N_under= np.nanstd(self.norm_x_mins_N_under)
        self.std_norm_x_maxs_N_under= np.nanstd(self.norm_x_maxs_N_under)  
        self.std_norm_err_N_under= np.nanstd(self.norm_err_N_under)
        self.std_norm_N_under= np.nanstd(self.norm_N_under)

        self.mean_norm_x_mins_N_over= np.nanmean(self.norm_x_mins_N_over)
        self.mean_norm_x_maxs_N_over= np.nanmean(self.norm_x_maxs_N_over)  
        self.mean_norm_err_N_over= np.nanmean(self.norm_err_N_over)
        self.mean_norm_N_over= np.nanmean(self.norm_N_over)

        self.std_norm_x_mins_N_over= np.nanstd(self.norm_x_mins_N_over)
        self.std_norm_x_maxs_N_over= np.nanstd(self.norm_x_maxs_N_over)  
        self.std_norm_err_N_over= np.nanstd(self.norm_err_N_over)
        self.std_norm_N_over= np.nanstd(self.norm_N_over)


    def get_parameters(self, par_name):
        par_standard, par_err_standard = [], []
        par_N_under, par_err_N_under = [], []
        par_N, par_err_N = [], []
        par_N_over, par_err_N_over = [], []
        
        
        for a in self.rnds:
            a = str(a)
            err = self.data[(a)]['result'][f'best_fit_{par_name}_error_standard']
            norm = self.data[(a)]['result'][f'best_fit_{par_name}_standard']
            par_standard.append(norm)
            par_err_standard.append(err)
            
            err = self.data[(a)]['result'][f'best_fit_{par_name}_error_N']
            norm = self.data[(a)]['result'][f'best_fit_{par_name}_N']
            par_N.append(norm)
            par_err_N.append(err)

           
            
            if self.false_est:
                err = self.data[(a)]['result'][f'best_fit_{par_name}_error_N_under']
                norm = self.data[(a)]['result'][f'best_fit_{par_name}_N_under']
                par_N_under.append(norm)
                par_err_N_under.append(err)
               
                err = self.data[(a)]['result'][f'best_fit_{par_name}_error_N_over']
                norm = self.data[(a)]['result'][f'best_fit_{par_name}_N_over']
                par_N_over.append(norm)
                par_err_N_over.append(err)
            else:
                par_N_under.append(np.nan)
                par_err_N_under.append(np.nan)
                par_N_over.append(np.nan)
                par_err_N_over.append(np.nan)
                
        mean_par_standard = np.mean(par_standard)
        mean_par_err_standard = np.mean(par_err_standard)
        mean_par_N_under = np.mean(par_N_under)
        mean_par_err_N_under = np.mean(par_err_N_under)
        mean_par_N = np.mean(par_N)
        mean_par_err_N = np.mean(par_err_N)
        mean_par_N_over = np.mean(par_N_over)
        mean_par_err_N_over = np.mean(par_err_N_over)
        tupel = (mean_par_standard, mean_par_err_standard, mean_par_N_under, mean_par_err_N_under, mean_par_N, mean_par_err_N, mean_par_N_over,mean_par_err_N_over)
        return tupel
    
        
    def success (self):
        self.success_standard  =[]
        self.success_N  =[]

        for a in self.rnds:
            try:
                self.success_standard.append(self.data[str(a)]['result']['success_standard'])
                self.success_N.append(self.data[str(a)]['result']['success_N'])
            except:
                self.success_standard.append(np.nan)
                self.success_N.append(np.nan)
                
   
        
    