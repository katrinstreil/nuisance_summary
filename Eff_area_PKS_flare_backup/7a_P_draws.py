# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Runnning with gammapy-dev/IRF_model
#

# %%

def save():
    with open("data/7aP_P_draw_info.txt", "a") as myfile:
        myfile.write(str(float(shift_rnd[0])) + '    '+ str(float(tilt_rnd[0])) + '    ' +  str(float(dataset.stat_sum())) + '\n')
    with open("data/7aP_P_draw_par.txt", "a") as myfile:
        myfile.write(stri + '\n')
    with open("data/7aP_P_draw_flux.txt", "a") as myfile:
        myfile.write( ff + '\n')
    with open("data/7aP_P_draw_flux2e.txt", "a") as myfile:
        myfile.write( ff2 + '\n')

    with open("data/7aP_N_P_draw_par.txt", "a") as myfile:
        myfile.write(stri_N + '\n')
    with open("data/7aP_N_P_draw_flux.txt", "a") as myfile:
        myfile.write( ffN + '\n')
    with open("data/7aP_N_P_draw_flux2e.txt", "a") as myfile:
        myfile.write( ffN2 + '\n')


import gammapy 
import pyximport
import numpy as np
import astropy.units as u
import sys
from gammapy.modeling import Fit, Parameter, Parameters
from gammapy.modeling.models import Models
from gammapy.maps import MapAxis
from gammapy.modeling.models.spectral import scale_plot_flux
from gammapy.estimators import  FluxPointsEstimator
from gammapy.modeling.models import IRFModels, EffAreaIRFModel, ERecoIRFModel
    
sys.path.append('/home/katrin/Documents/nuisance_summary/')
sys.path.append('../')
import Dataset_load 
from  Dataset_Setup import Setup, GaussianCovariance_matrix

pyximport.install()

print(f'loaded gammapy version: {gammapy.__version__} ' )
print('Supposed to be 1.0 (21-12-2022)' )
livetimes = np.logspace(-2, 2,  13)[:6] 
livetimes = np.append(livetimes, np.logspace(0, 2, 7))
livetimes= livetimes[7:8]
for live in livetimes:
#live = 0.1

    dataset_asimov = Dataset_load.create_asimov(
        model="crab", source="PKSflare", parameters=None,
        livetime = f"{live}-hr"
    )

    mask = dataset_asimov.mask.data.sum(axis=2).sum(axis=1)>0

    ebins = dataset_asimov.counts.geom.axes[0].center[mask]
    print(len(ebins))


    norm = 0.1
    tilt = 0.025
    bias = 0.0
    resolution = 0.0

    N = 1

    save_flux = True
    save_fluxpoints = 1
    save_fluxpoints_N = 1
    dataset_N = True


    for n in range(N):
        print()
        print('====' * 30)
        print(n)
        print('====' * 30)
        res_rnd = [0.] 
        bias_rnd =  [0.] 
        shift_rnd = np.random.normal(0, norm, 1)
        tilt_rnd = np.random.normal(0, tilt, 1)

        print(f"shift {shift_rnd}, tilt {tilt_rnd}")
        setup = Setup(dataset_input=dataset_asimov, rnd = True)
        #setup.set_up_irf_sys(bias, resolution, norm, tilt)
        dataset, dataset_N = setup.run()
        # irf model
        setup.set_irf_model(dataset_N)
        dataset_N.models.parameters['resolution'].frozen = True
        dataset_N.models.parameters['bias'].frozen = True

        dataset_N.irf_model.parameters['tilt'].frozen = False
        dataset_N.irf_model.parameters['norm'].frozen = False
        setup.set_irf_prior(dataset_N, bias_rnd[0], res_rnd[0], shift_rnd, tilt_rnd)

        fit_cor = Fit(store_trace=False)
        result_cor = fit_cor.run([dataset])


        stri = ""
        for p in ['amplitude', 'index', 'lambda_', 'norm', 'tilt']:
            stri += str(dataset.models.parameters[p].value)  + '   ' +  str(dataset.models.parameters[p].error)  + '   '
        stri += str(live) + "  "
        print(stri)


        fluxes = []
        for e in ebins:
            flux =  dataset.models[0].spectral_model(e)
            fluxes.append(flux.value)

        ff = str()
        for f in fluxes:
            ff += str(f) + "  "
        #print(ff)

        energy_bounds = (ebins[0], ebins[-1] ) * u.TeV

        energy_min, energy_max = energy_bounds
        energy = MapAxis.from_energy_bounds(
            energy_min,
            energy_max,
            len(ebins),
        )

        fluxe2, _ = dataset.models[0].spectral_model._get_plot_flux(sed_type='dnde', energy=energy)
        fluxe2 = scale_plot_flux(fluxe2, energy_power=2)
        fluxe2 = fluxe2.quantity[:, 0, 0]
        fluxe2 = np.array(fluxe2)   
        ff2 = str()
        for f in fluxe2:
            ff2 += str(f) + "  "

        energy_edges = dataset.geoms['geom'].axes[0].edges[::2]

        fit_cor = Fit(store_trace=False)
        result_cor = fit_cor.run([dataset_N])

        stri_N = ""
        for p in ['amplitude', 'index', 'lambda_', 'norm', 'tilt', 'bias', 'resolution']:
            stri_N += str(dataset_N.models.parameters[p].value)  + '   ' +  str(dataset_N.models.parameters[p].error)  + '   '
        stri_N += str(live) + "  "
        print(stri_N)

        fluxes = []
        for e in ebins:
            flux =  dataset_N.models[0].spectral_model(e)
            fluxes.append(flux.value)

        ffN = str()
        for f in fluxes:
            ffN += str(f) + "  "

        energy_bounds = (ebins[0], ebins[-1] ) * u.TeV

        energy_min, energy_max = energy_bounds
        energy = MapAxis.from_energy_bounds(
            energy_min,
            energy_max,
            len(ebins),
        )

        fluxe2, _ = dataset_N.models[0].spectral_model._get_plot_flux(sed_type='dnde', energy=energy)
        fluxe2 = scale_plot_flux(fluxe2, energy_power=2)
        fluxe2 = fluxe2.quantity[:, 0, 0]
        fluxe2 = np.array(fluxe2)   
        ffN2 = str()
        for f in fluxe2:
            ffN2 += str(f) + "  "


        if save_fluxpoints:
            dataset.models.parameters['amplitude'].scan_n_sigma  = 5
            dataset_N.models.parameters['amplitude'].scan_n_sigma  = 5
            
            esti  = FluxPointsEstimator(energy_edges= energy_edges, selection_optional = "all")
            fluxpoints = esti.run([dataset])
            fluxpoints_N = esti.run([dataset_N])
            fluxpoints_N.write(f'data/fluxpoints/6P_fluxpoints_N_{live}_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits')
            Models([dataset_N.models[0]]).write(f'data/fluxpoints/6P_model_N_{live}_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.yaml')
            fluxpoints.write(f'data/fluxpoints/6P_fluxpoints_{live}_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits')
            Models([dataset.models[0]]).write(f'data/fluxpoints/6P_model_{live}_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.yaml')

        save()

        import matplotlib.pyplot as plt

        ep = 2
        ax = dataset.models[0].spectral_model.plot((0.1,100)*u.TeV, color = 'tab:blue',
                                                  label = "without nui",
                                                  energy_power = ep)


        dataset_asimov.models[0].spectral_model.plot((0.1,100)*u.TeV,ax = ax, color = 'black',
                                                    energy_power = ep)

        dataset_N.models[0].spectral_model.plot((0.1,100)*u.TeV,ax = ax, color = 'tab:orange',
                                               label = "with nui",
                                               energy_power = ep)
        dataset_N.models[0].spectral_model.plot_error((0.1,100)*u.TeV,ax = ax, facecolor = 'tab:orange',
                                                     energy_power = ep)
        dataset.models[0].spectral_model.plot_error((0.1,100)*u.TeV,ax = ax, facecolor = 'tab:blue',
                                                   energy_power = ep)

        fluxpoints_N.plot(ax =ax,energy_power = ep )
        fluxpoints.plot(ax =ax, energy_power = ep)
        ax.legend(title = f"live: {live} hr norm:{shift_rnd[0]:.3} tilt:{tilt_rnd[0]:.3}")
        fig = plt.gcf()
        fig.savefig(f"data/plots/{live}_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.png")
        plt.close()




# %%
dataset_N

# %%
