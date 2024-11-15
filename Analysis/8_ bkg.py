"""
Runnning with gammapy-dev/IRF_model
-----------------------------------

Fitting asimov datasets with nuisance parameters based on the different
livetimes

"""


######################################################################
# Setup
# ~~~~~
# 

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from gammapy.maps import Map
from astropy.coordinates import SkyCoord, Angle
from gammapy.modeling import Fit,  Parameters, Covariance , Parameter
from gammapy.datasets import MapDataset ,Datasets, FluxPointsDataset
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    SkyModel,
    PointSpatialModel,
    PowerLawNormSpectralModel,
    Models,
    SpatialModel,
    FoVBackgroundModel,
    PiecewiseNormSpectralModel,
)
from gammapy.estimators import TSMapEstimator, ExcessMapEstimator
from gammapy.estimators import FluxPoints, FluxPointsEstimator
from scipy.interpolate import interp2d

from regions import CircleSkyRegion, RectangleSkyRegion
import yaml
import sys
sys.path.append('../')
import Dataset_load 

from  Dataset_Setup import Setup, GaussianCovariance_matrix

from matplotlib import rc
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)



######################################################################
# Read dataset
# ------------
# 

c = Dataset_load.load_config()
awo, aw, ewo, ew = c['_colors']  


livetime = 100.0 # c['livetime']
zero = c['zero'] 
norm = c['norm'] 
tilt = c['tilt'] 
bias = c['bias'] 
resolution = c['resolution'] 
magnitude = c['magnitude'] 
corrlength = c['corrlength']
sys = c['sys'] 
folder = c['folder']
parameter_names = c['parameter_names']        
nbidx = 0
print(livetime)
print(sys)

c['model']

# %%time
amplitude= Parameter('amplitude', value = 3.85e-12, unit=u.Unit("1 / (TeV s cm2)"))
dataset_input_point  = Dataset_load.create_asimov(model = c['model'], source = c['source'], 
                                               livetime = f"{livetime}-hr",
                                        parameters = [amplitude],
                                           spatial_model =None)
from gammapy.modeling.models import GaussianSpatialModel
gaussian = GaussianSpatialModel(lon_0 = dataset_input_point.models.parameters['lon_0'],
                                lat_0 = dataset_input_point.models.parameters['lat_0'],
                                sigma = 0.3 *u.deg
                               )


dataset_input  = Dataset_load.create_asimov(model = c['model'], source = c['source'], 
                                               livetime = f"{livetime}-hr",
                                        parameters = [amplitude],
                                           spatial_model =gaussian)


dataset_input.npred().sum_over_axes().plot(add_cbar =1)

    
setup = Setup(dataset_input=dataset_input)
#setup.set_up_irf_sys(bias, resolution, norm, tilt)
dataset_asimov, dataset_asimov_N = setup.run()
# irf model
#setup.set_irf_model(dataset_asimov_N)
if "Bkg" in sys or "BKG" in sys:

    # piece wise model
    # remove old bkg model
    setup.set_up_bkg_sys_V( breake = 3,
                        index1 = 8,
                        index2 = 1.5, 
                        magnitude = magnitude )

    dataset_asimov, dataset_asimov_N = setup.run()

    #setup.unset_model(dataset_asimov_N, FoVBackgroundModel)
    #setup.set_piecewise_bkg_model(dataset_asimov_N)
    # energy of the following parameters smaller than ethrshold
    dataset_asimov_N.background_model.parameters['norm0'].frozen = True
    dataset_asimov_N.background_model.parameters['norm1'].frozen = True
    dataset_asimov_N.background_model.parameters['norm2'].frozen = True
    dataset_asimov_N.background_model.parameters['norm3'].frozen = True
    print("magnitude", magnitude, "corrlength", corrlength)
    setup.set_bkg_prior(dataset_asimov_N, magnitude= magnitude, corrlength= corrlength)
    frozen_pos = 1
    if frozen_pos:
        dataset_asimov.models.parameters['lon_0'].frozen = True
        dataset_asimov.models.parameters['lat_0'].frozen = True
        dataset_asimov_N.models.parameters['lon_0'].frozen = True
        dataset_asimov_N.models.parameters['lat_0'].frozen = True
    else:
        for d in [dataset_asimov, dataset_asimov_N]:
            delta = 0.01 
            dataset_asimov.models.parameters['lon_0'].min = dataset_asimov.models.parameters['lon_0'].value - delta
            dataset_asimov.models.parameters['lon_0'].max = dataset_asimov.models.parameters['lon_0'].value + delta 
            dataset_asimov.models.parameters['lat_0'].min = dataset_asimov.models.parameters['lat_0'].value - delta
            dataset_asimov.models.parameters['lat_0'].max = dataset_asimov.models.parameters['lat_0'].value + delta
                
         
    
    
ax = setup.dataset_helper.background_model.spectral_model.plot(color= 'black')

ax.set_yscale("linear")
setup.dataset_helper.background_model.spectral_model.parameters.value

dataset_asimov_N.background_model.evaluate(1*u.TeV)

setup.dataset_helper.background_model.evaluate(1*u.TeV)

setup.dataset_helper.counts.sum_over_axes().plot(add_cbar = 1)

setup.dataset_helper.npred().sum_over_axes().plot(add_cbar = 1)

setup.dataset_helper.npred_background().sum_over_axes().plot(add_cbar = 1)

setup.dataset_helper.plot_residuals()

dataset_asimov_N.counts.sum_over_axes().plot(add_cbar = 1)

dataset_asimov_N.npred().sum_over_axes().plot(add_cbar = 1)

dataset_asimov_N.npred_background().sum_over_axes().plot(add_cbar = 1)

dataset_asimov_N.plot_residuals()


# %%time
fitting = 0
if fitting:
    fit_cor = Fit(store_trace=0)
    minuit_opts = {"tol": 0.1, "strategy": 2}
    fit_cor.backend = "minuit"
    fit_cor.optimize_opts = minuit_opts
    result_cor = fit_cor.run(dataset_asimov)
    print(result_cor)
    print("saving")
    path = f'../{folder}/data/0_model_livetime_{livetime}.yml'
    dataset_asimov.models.write(path,overwrite=True)
    
    
else:
    path = f'../{folder}/data/0_model_livetime_{livetime}.yml'
    dataset_asimov.models = Models.read(path)
    print(path[:-4])
    cov = dataset_asimov.models.read_covariance(path = path[:-4]+"_covariance.dat", filename = "",
                                            format="ascii.fixed_width")


# %%time
fitting = 0
if fitting:
    fit_cor = Fit(store_trace=0)
    result_cor = fit_cor.run(dataset_asimov_N)
    print(result_cor)
    print("saving")
    path = f'../{folder}/data/0_model_nui_livetime_{livetime}.yml'
    dataset_asimov_N.models.write(path,overwrite=True)
    
    
else:
    path = f'../{folder}/data/0_model_nui_livetime_{livetime}.yml'
    dataset_asimov_N.models = Models.read(path)
    dataset_asimov_N.background_model.parameters['_norm'].value = 1.

print(dataset_asimov_N.background_model)

from scipy.linalg import inv
cov = inv(dataset_asimov_N.background_model.parameters.prior[1].covariance_matrix)
cov[-1]

import colors as l
aw = l.purple
awo =l.orange

labelw = "Fit with background systematic"
labelwo = "Fit without background systematic"



label = [ 'input', 'dataset_asimov', 'dataset_asimov_N']
for i, d in enumerate([ setup.dataset_helper, dataset_asimov, dataset_asimov_N]):
    print(label[i])

    for p in ['amplitude','index', 'lambda_','sigma']:
        par =d.models.parameters[p]
        factor = 1 
        if p == 'amplitude':
            factor = 1e12
        print(f" {p}: ${par.value*factor :.3}  \pm {par.error*factor :.3} $")
    print()

models_no  = [dataset_asimov.models]
models  = [dataset_asimov_N.models]

headline = "\\textbf{Extended Source} &   $\Phi_0\ [10^{-12}\, \\text{cm}^{-2}\, \\text{s}^{-1}\,\\text{TeV}^{-1}]$  & $\Gamma$ & $\lambda = 1/E_c\ [\\text{TeV}^{-1}]$ & $\sigma\, [\\text{deg}]$   \\\  \hline "
input_ = '  Simulation input & $3.85 $  & $2.30 $  & $0.10 $ & 0.3  \\\  \hline'

par = models_no[0].parameters['amplitude']
without = rf' Without systematic & \errorsym {{{par.value*1e12:.3}}} {{{par.error*1e12:.3}}} '
par = models_no[0].parameters['index']
without += rf' & \errorsym {{{par.value:.3}}} {{{par.error:.3}}}   '
par = models_no[0].parameters['lambda_']
without += rf' & \errorsym  {{{par.value:.3}}} {{{par.error:.3}}}   '
par = models_no[0].parameters['sigma']
without += rf' & \errorsym  {{{par.value:.3}}} {{{par.error:.3}}}    \\  \hline'


par = models[0].parameters['amplitude']
with_ = rf' Fitting bkg. sys.   & \errorsym {{{par.value*1e12:.3}}} {{{par.error*1e12:.3}}}  '
par = models[0].parameters['index']
with_ += rf' & \errorsym  {{{par.value:.3}}} {{{par.error:.3}}}   '
par = models[0].parameters['lambda_']
with_ += rf' & \errorsym  {{{par.value:.3}}} {{{par.error:.3}}}'
par = models[0].parameters['sigma']
with_ += rf' & \errorsym  {{{par.value:.3}}} {{{par.error:.3}}}   \\  \hline'

print(headline)
print(input_)

print(without)
print(with_)



######################################################################
# Scan
# ----
# 


parameter_names_1  = ['amplitude', 'index', 'lambda_', 'sigma']

def computing_scan(dataset, note):
        
    fit_cor = Fit(store_trace=False)
    result_cor = fit_cor.run(dataset)
    print(dataset_asimov.models)
    
    results = []
    for parname1 in parameter_names_1 :
        if True: #parname1 == 'lambda_':
            print("scanning",  parname1)
            dataset.models.parameters[parname1].scan_n_values=numpoints
            result = fit_cor.stat_profile(dataset,
                                 dataset.models.parameters[parname1],
                                reoptimize = True
                                )

            contour_write = dict()
            for k in result.keys():
                print(k)
                if k != "fit_results":
                    contour_write[k] = [float(_) for _ in result[k]]#.tolist()
            print(contour_write)
            with open(f"../{c['folder']}/data/4_scan_{note}_{parname1}_{numpoints}_{livetime}.yml", "w") as outfile:
                yaml.dump(contour_write, outfile, default_flow_style=False)

            results.append(result)
    return results
        
def read_in_scan(note):
    results = []
    for parname1 in parameter_names_1 :
        try:
            with open(f"../{c['folder']}/data/4_scan_{note}_{parname1}_{numpoints}_{livetime}.yml", "r") as stream:
                contour = yaml.safe_load(stream)
        except:
            with open(f"../{c['folder']}/data/4_scan_{note}_{parname1}_{livetime}.yml", "r") as stream:
                contour = yaml.safe_load(stream)
        results.append(contour)
    return results


# %%time
numpoints = 20
computing = 0
if computing:
    results = computing_scan(dataset_asimov, "2.15h")
else:
    results = read_in_scan("2.15h")
    path = f'../{folder}/data/0_model_livetime_{livetime}.yml'
    dataset_asimov.models = Models.read(path)
for i, r in enumerate(results):
    print(r)
    fig,ax = plt.subplots(1,1, figsize = (3,2))
    plt.plot(r[list(r.keys())[0]], r['stat_scan'] - np.min(r['stat_scan']))

    if i == 0 :
        ax.set_xscale('log')
    ax.errorbar(x = dataset_asimov.models.parameters[parameter_names_1[i]].value,
               y = 1,
               xerr = dataset_asimov.models.parameters[parameter_names_1[i]].error,
               fmt = 'x')

# %%time
computing = 1
numpoints = 20

if computing:
    dataset_asimov_N.models.parameters['lon_0'].frozen = True
    dataset_asimov_N.models.parameters['lat_0'].frozen = True
    
    results_N = computing_scan(dataset_asimov_N, "N_2.15h")
else:
    results_N = read_in_scan("N_2.15h")
    try:
        path = f'../{folder}/data/0_model_nui_livetime_{livetime}.yml'
        dataset_asimov_N = Dataset_load.load_dataset_N(dataset_asimov_N, path,bkg_sys = False)        
    except:
        path = f'../{folder}/data/0_model_nui_livetime_{livetime}_1000.yml'
        dataset_asimov_N = Dataset_load.load_dataset_N(dataset_asimov_N, path,bkg_sys = False)        
        
print(results_N)

results_N  = results



import upper_limit_18_02
import colors as s
colors_ = [s.blue, s.orange,
          s.lblue, s.lorange]

#colors_ = [awo[0] , aw[0],
#           awo[1] , aw[1]]

for i, p in enumerate(parameter_names_1):
    #if p == 'index':
    if True:
        stat_profile_N = results_N[i]
        stat_profile = results[i]

        stat_profile_N['stat_scan'] -= np.min(stat_profile_N['stat_scan'])
        stat_profile['stat_scan'] -= np.min(stat_profile['stat_scan'])

        fig = plt.figure()
        ll_N_a = stat_profile_N.copy()
        case = 'spectral'
        if p == 'sigma':
            case = 'spatial'
            
        amplitude_err = dataset_asimov.models[0].parameters[p].error
        amplitude = dataset_asimov.models[0].parameters[p].value
        amplitude_err_N = dataset_asimov_N.models[0].parameters[p].error
        amplitude_N = dataset_asimov_N.models[0].parameters[p].value

        fig, ax = plt.subplots(1,1)
            
        ### LIKELIHOOD
        # scan
        ll_a = stat_profile.copy()
        ul_a = upper_limit_18_02.upper_limit(ll_a,0,0,  name=f'{dataset_asimov.models[0].name}.{case}.{p}_scan')
        L_a, x_a = ul_a.interpolate()
        plt.plot(x_a, L_a(x_a),label = "-2log (L)", linestyle = 'dashed', color = colors_[0])
    
        ylim= ax.get_ylim()
        ymax = 2#ylim[1]
        min_, er_neg, er_pos = ul_a.likelihood_error_asymmetric()
        min_ = min_[0]; er_neg = er_neg[0]; er_pos = er_pos[0]; 
        dataset_asimov.models.parameters[p].error_n = er_neg
        dataset_asimov.models.parameters[p].error_p = er_pos
        
        factor = 1
        if p == 'amplitude':
            factor = 1e11

        ax.fill_between(  [min_-er_neg, min_+ er_pos], ylim[0], ymax, alpha = 0.5, color=colors_[2],
                        label = f'1$\sigma$ error (Scan): -{er_neg*factor:.2} +{er_pos*factor:.2} ')
     
        
        ax.vlines(amplitude-amplitude_err, ylim[0], ymax, color = colors_[0], linestyle ='dotted')
        ax.vlines(amplitude+amplitude_err, ylim[0], ymax, color = colors_[0], linestyle ='dotted',
                 label =  f'1$\sigma$ error (Minuit): {amplitude_err*factor:.2}')

           
        ### POSTERIOR
        # scan
        ul_N_a = upper_limit_18_02.upper_limit(ll_N_a,0,0, 
                                               name=f'{dataset_asimov.models[0].name}.{case}.{p}_scan')
        L_N_a, x_N_a = ul_N_a.interpolate()
        plt.plot(x_N_a, L_N_a(x_N_a),label = "-2log (P)", color = colors_[1])
        
        
        min_N, er_negN, er_posN = ul_N_a.likelihood_error_asymmetric()
        min_N = min_N[0]; er_negN = er_negN[0]; er_posN = er_posN[0]; 
        dataset_asimov_N.models.parameters[p].error_n = er_negN
        dataset_asimov_N.models.parameters[p].error_p = er_posN


        ax.fill_between(  [min_N-er_negN, min_N+ er_posN], ylim[0], ymax, alpha = 0.5, color = colors_[3],
                        label = f'1$\sigma$ error (Scan): -{er_negN*factor:.5} +{er_posN*factor:.5} ')
        ax.vlines(amplitude_N-amplitude_err_N, ylim[0], ymax,color = colors_[1] ,
                  linestyles='dotted'
                 )
        ax.vlines(amplitude_N+amplitude_err_N, ylim[0], ymax,color = colors_[1],
                  linestyles='dotted',
                    label = f'1$\sigma$ error (Minuit): $\pm${amplitude_err_N*factor:.5}')
                 
        nn = 2
        #ax.set_xlim(amplitude_N-amplitude_err_N*nn, 
        #           amplitude_N+amplitude_err_N*nn)
        ax.set_ylim(np.min(stat_profile['stat_scan'])-0.5,
                    np.min(stat_profile['stat_scan'])+ 3)

        
        
       
    
    
        xx = ax.get_xlim()
        alpha = 0.6
        ax.hlines(0, xx[0], xx[1], color = 'grey', alpha = alpha)
        ax.hlines(1, xx[0], xx[1], color = 'grey', alpha = alpha)
        if p == 'amplitude':
            str_= "[$\\mathrm{TeV^{-1}\\,s^{-1}\\,cm^{-2}}$]"
            plt.xlabel(f"Source strength " + str_) 
        else:
            plt.xlabel(p)
        plt.ylabel("-2log (L) [arb. unit]")
        plt.legend(ncol = 2)

    fig.savefig(f"../{c['folder']}/plots/4_scan_{p}.pdf")
    
    

headline = "\\textbf{Extended Source} & &  $\Phi_0\ [10^{-11}\, \\text{cm}^{-2}\, \\text{s}^{-1}\,\\text{TeV}^{-1}]$  & $\Gamma$ & $\lambda = 1/E_c\ [\\text{TeV}^{-1}]$ & $\sigma\, [\text{deg}]$ &  \\\  \hline "
input_ = ' & Simulation input & $3.85 $  & $2.30 $  & $0.10 $ & 0.3  \\\  \hline'

par = models_no[0].parameters['amplitude']
without = rf' &Without systematic & \error {{{par.value*1e11:.3}}} {{{par.error*1e11:.3}}}  {{{par.error_p*1e11:.3}}}  {{{par.error_n*1e11:.3}}} '
par = models_no[0].parameters['index']
without += rf' & \error {{{par.value:.3}}} {{{par.error:.3}}}  {{{par.error_p:.3}}}  {{{par.error_n:.3}}} '
par = models_no[0].parameters['lambda_']
without += rf' & \error  {{{par.value:.3}}} {{{par.error:.3}}}  {{{par.error_p:.3}}}  {{{par.error_n:.3}}}  \\  \hline'
par = models_no[0].parameters['sigma']
without += rf' & \error  {{{par.value:.3}}} {{{par.error:.3}}}  {{{par.error_p:.3}}}  {{{par.error_n:.3}}}  \\  \hline'


par = models[0].parameters['amplitude']
eff = rf' Effective Area & With fitting  & \error {{{par.value*1e11:.3}}} {{{par.error*1e11:.3}}}  {{{par.error_p*1e11:.3}}}  {{{par.error_n*1e11:.3}}} '
par = models[0].parameters['index']
eff += rf' & \error  {{{par.value:.3}}} {{{par.error:.3}}}  {{{par.error_p:.3}}}  {{{par.error_n:.3}}} '
par = models[0].parameters['lambda_']
eff += rf' & \error  {{{par.value:.3}}} {{{par.error:.3}}}  {{{par.error_p:.3}}}  {{{par.error_n:.3}}}  \\  \hline'
par = models[0].parameters['sigma']
eff += rf' & \error  {{{par.value:.3}}} {{{par.error:.3}}}  {{{par.error_p:.3}}}  {{{par.error_n:.3}}}  \\  \hline'






