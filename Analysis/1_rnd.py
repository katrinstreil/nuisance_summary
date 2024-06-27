def save():
    with open(f"{path}/data/1_P_draw_info.txt", "a") as myfile:
        info = str(float(shift_rnd[0])) + '    '+ str(float(tilt_rnd[0])) + '    '
        info += str(float(bias_rnd[0])) + '    '+ str(float(res_rnd[0])) + '    '
        info +=  str(float(dataset.stat_sum())) + '\n'
        myfile.write(info)
    with open(f"{path}/data/1_P_draw_par.txt", "a") as myfile:
        myfile.write(stri + '\n')
    with open(f"{path}/data/1_P_draw_flux.txt", "a") as myfile:
        myfile.write( ff + '\n')
    with open(f"{path}/data/1_P_draw_flux2e.txt", "a") as myfile:
        myfile.write( ff2 + '\n')

    with open(f"{path}/data/1_N_P_draw_par.txt", "a") as myfile:
        myfile.write(stri_N + '\n')
    with open(f"{path}/data/1_N_P_draw_flux.txt", "a") as myfile:
        myfile.write( ffN + '\n')
    with open(f"{path}/data/1_N_P_draw_flux2e.txt", "a") as myfile:
        myfile.write( ffN2 + '\n')
        
def computing_contour(dataset, note):
        
    results = []
    for parname1, parname2 in parameter_names :
        numpoints = 5
        print( parname1, parname2, numpoints)
        result = fit_cor.stat_contour(dataset,
                             dataset.models.parameters[parname1],
                             dataset.models.parameters[parname2],
                              numpoints=numpoints 
                                      
                            )

        contour_write = dict()
        for k in result.keys():
            print(k)
            if k != "success":
                contour_write[k] = result[k].tolist()
        import yaml
        with open(f"{path}/data/contours/{note}_{parname1}_{parname2}.yml", "w") as outfile:
            yaml.dump(contour_write, outfile, default_flow_style=False)
        



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


c = Dataset_load.load_config()
awo, aw, ewo, ew = c['_colors']  

livetimes = c['livetimes']
livetime = c['livetime']
sys = c['sys']
norm = c['norm'] 
tilt = c['tilt'] 
bias =  c['bias'] 
resolution = c['resolution'] 
path = f"../{c['folder']}"
parameter_names = c['parameter_names']        

#for live in livetimes[8:]:
for live in [livetime]:

    dataset_asimov = Dataset_load.create_asimov(
        model=c['model'], source=c['source'], parameters=None,
        livetime = f"{live}-hr"
    )

    mask = dataset_asimov.mask.data.sum(axis=2).sum(axis=1)>0
    ebins = dataset_asimov.counts.geom.axes[0].center[mask]


    N = 1000
    save_flux = True
    save_fluxpoints = 0
    save_fluxpoints_N = 0
    dataset_N = True
    contour = 0
    zero_sys =0


    for n in range(N):
        print()
        print('====' * 30)
        print(n)
        print('====' * 30)
        res_rnd = np.random.normal(0, resolution, 1)
        bias_rnd =  np.random.normal(0, bias, 1)
        shift_rnd = np.random.normal(0, norm, 1)
        tilt_rnd = np.random.normal(0, tilt, 1)
        if zero_sys:
            nn = 71 #np.random.randint(0,100)
            print("nn", nn)
            rnd = nn
        else:
            rnd = "True"
        
        if zero_sys:
            shift_rnd, tilt_rnd = np.array([0.]), np.array([0.])
            bias_rnd, res_rnd = np.array([0.]), np.array([0.])
        
        print(f"shift {shift_rnd}, tilt {tilt_rnd},  bias {bias_rnd}, res {res_rnd}")
        setup = Setup(dataset_input=dataset_asimov, rnd = rnd)
        setup.set_up_irf_sys(bias_rnd, res_rnd, shift_rnd, tilt_rnd)

        dataset, dataset_N = setup.run()
        # irf model
        # happens in set_up_irf_sys
        # setup.set_irf_model(dataset_N)
        if  "Eff_area" in sys:
            dataset_N.models.parameters['resolution'].frozen = True
            dataset_N.models.parameters['bias'].frozen = True
            dataset_N.irf_model.parameters['tilt'].frozen = False
            dataset_N.irf_model.parameters['norm'].frozen = False
            dataset_N.e_reco_n = 10
            setup.set_irf_prior(dataset_N, bias, resolution, norm, tilt)
            print(dataset_N.irf_model)
            
        if sys == "E_reco":
            dataset_N.models.parameters['resolution'].frozen = True
            dataset_N.models.parameters['bias'].frozen = False
            dataset_N.irf_model.parameters['tilt'].frozen = True
            dataset_N.irf_model.parameters['norm'].frozen = True
            setup.set_irf_prior(dataset_N, bias, resolution, norm, tilt)
        
        if sys == "Combined":
            dataset_N.models.parameters['resolution'].frozen = True
            dataset_N.models.parameters['bias'].frozen = False
            dataset_N.irf_model.parameters['tilt'].frozen = False
            dataset_N.irf_model.parameters['norm'].frozen = False
            setup.set_irf_prior(dataset_N, bias, resolution, norm, tilt)
            
        if sys == "BKG":
            magnitude = c['magnitude']
            corrlength = c['corrlength']
            # piece wise model
            # remove old bkg model
            setup.set_up_bkg_sys_V( breake = 10,
                                index1 = 2,
                                index2 = 1.5, 
                                magnitude = magnitude )

            dataset_asimov, dataset_asimov_N = setup.run()

            setup.unset_model(dataset_asimov_N, FoVBackgroundModel)
            setup.set_piecewise_bkg_model(dataset_asimov_N)
            # energy of the following parameters smaller than ethrshold
            dataset_asimov_N.background_model.parameters['norm0'].frozen = True
            dataset_asimov_N.background_model.parameters['norm1'].frozen = True
            dataset_asimov_N.background_model.parameters['norm2'].frozen = True
            dataset_asimov_N.background_model.parameters['norm3'].frozen = True
            setup.set_bkg_prior(dataset_asimov_N, magnitude, corrlength)
            frozen_pos = 1
            if frozen_pos:
                dataset_asimov.models.parameters['lon_0'].frozen = True
                dataset_asimov.models.parameters['lat_0'].frozen = True
                dataset_asimov_N.models.parameters['lon_0'].frozen = True
                dataset_asimov_N.models.parameters['lat_0'].frozen = True
        
        fit_cor = Fit(store_trace=False)
        dataset.plot_residuals()
        result_cor = fit_cor.run([dataset])
        print("fit w/o nui ended:")
        print(result_cor)
        print(dataset.models)

        stri = ""
        parameters =  ['amplitude', 'index', 'lambda_', 'norm', 'tilt']
        if "crab_break" in c['model']:
            parameters =  ['amplitude', 'index1', 'index2', 'ebreak', 'beta', 'norm', 'tilt']
        if "crab_log" in c['model']:
            parameters =  ['amplitude', 'alpha', 'beta', 'norm', 'tilt']
            
        for p in parameters:
            stri += str(dataset.models.parameters[p].value)  + '   ' +  str(dataset.models.parameters[p].error)  + '   '
        stri += str(live) + "  "
        print(stri)


     
        energy_edges = dataset.geoms['geom'].axes[0].edges#[::2]
        energy_bounds = (energy_edges[0], energy_edges[-1] ) #* u.TeV

        energy_min, energy_max = energy_bounds
        energy = MapAxis.from_energy_bounds(
            energy_min,
            energy_max,
            len(energy_edges),
            node_type='center'
        )
        fluxe2, _ = dataset.models[0].spectral_model._get_plot_flux(sed_type='dnde', energy=energy)
        fluxe2 = scale_plot_flux(fluxe2, energy_power=2)
        fluxe2 = fluxe2.quantity[:, 0, 0]
        fluxe2 = np.array(fluxe2)   
        ff2 = str()
        for f in fluxe2:
            ff2 += str(f) + "  "

        fluxes = []
        for e in energy.center:
            flux =  dataset.models[0].spectral_model(e)
            fluxes.append(flux.value)

        ff = str()
        for f in fluxes:
            ff += str(f) + "  "
        #print(ff)
        fit_cor = Fit(store_trace=False)
        result_cor = fit_cor.run([dataset_N])
        print()
        print("fit with nui ended:")
        print(result_cor)
        print(dataset_N.models)


        stri_N = ""
        [parameters.append(p) for p in ['norm', 'tilt', 'bias', 'resolution']]
        for p in parameters:
            stri_N += str(dataset_N.models.parameters[p].value)  + '   ' +  str(dataset_N.models.parameters[p].error)  + '   '
        stri_N += str(live) + "  "
        print(stri_N)

        fluxes = []
        for e in energy.center:
            flux =  dataset_N.models[0].spectral_model(e)
            fluxes.append(flux.value)

        ffN = str()
        for f in fluxes:
            ffN += str(f) + "  "

 
        fluxe2, _ = dataset_N.models[0].spectral_model._get_plot_flux(sed_type='dnde', energy=energy)
        fluxe2 = scale_plot_flux(fluxe2, energy_power=2)
        fluxe2 = fluxe2.quantity[:, 0, 0]
        fluxe2 = np.array(fluxe2)   
        ffN2 = str()
        for f in fluxe2:
            ffN2 += str(f) + "  "

        rnds = f"{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}_{bias_rnd[0]:.6}_{res_rnd[0]:.6}"
        if save_fluxpoints:
            print("computing Fluxpoints")
            dataset.models.parameters['amplitude'].scan_n_sigma  = 5
            dataset_N.models.parameters['amplitude'].scan_n_sigma  = 5

            esti  = FluxPointsEstimator(energy_edges= energy_edges[::2], 
                                        selection_optional =  "all",
                                        norm_min=0.2,
                                    norm_max=500,
                                    norm_n_values=15,
                                       )
            fluxpoints = esti.run([dataset])
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fluxpoints.plot()
            
            # freeze all but IRF for fp and reopt = True
            dataset_N.models[0].parameters.freeze_all()
            dataset_N.models[0].parameters['amplitude'].frozen = False
            dataset_N.background_model.parameters.freeze_all()
            esti  = FluxPointsEstimator(energy_edges= energy_edges[::2], selection_optional =[ "ul"],# "errn-errp", "all",
                                        norm_min=0.2,
                                    norm_max=500,
                                    norm_n_values=15,
                                       reoptimize=True)
            fluxpoints_N = esti.run([dataset_N])
            fluxpoints_N.plot(ax = ax)
            ax.legend(title = f"live: {live:.3} hr\n norm:{shift_rnd[0]:.3}\n tilt:{tilt_rnd[0]:.3}\n bias:{bias_rnd[0]:.3}")
            fig.savefig(f"{path}/data/fluxpoints/plots/{live}_{rnds}_{nn}.png")
            
            fluxpoints_N.write(f'{path}/data/fluxpoints/1P_fluxpoints_N_{live}_{rnds}_{nn}.fits',
                              overwrite = True)
            dataset_N.models.write(f'{path}/data/fluxpoints/1P_model_N_{live}_{rnds}_{nn}.yaml',
                                  overwrite = True)
            fluxpoints.write(f'{path}/data/fluxpoints/1P_fluxpoints_{live}_{rnds}_{nn}.fits',
                            overwrite = True)
            dataset.models.write(f'{path}/data/fluxpoints/1P_model_{live}_{rnds}_{nn}.yaml',
                                overwrite = True)
            with open(f"{path}/data/fluxpoints/1P_draw_fluxpoints.txt", "a") as myfile:
                myfile.write(str(nn) + '\n')
        if contour:
            computing_contour(dataset, rnds)
            print("N")
            computing_contour(dataset_N, "N"+rnds)
            with open(f"{path}/data/contours/1_P_draw_info.txt", "a") as myfile:
                info = str(float(shift_rnd[0])) + '    '+ str(float(tilt_rnd[0])) + '    '
                info += str(float(bias_rnd[0])) + '    '+ str(float(res_rnd[0])) + '    '
                info +=  str(float(dataset.stat_sum())) + '\n'
                myfile.write(info)
            with open(f"{path}/data/contours/1_P_draw_par.txt", "a") as myfile:
                myfile.write(stri + '\n')

            with open(f"{path}/data/contours/1_N_P_draw_par.txt", "a") as myfile:
                myfile.write(stri_N + '\n')

        if zero_sys == False and contour ==False: # else only the fluxpoints and models are saved but not the info
            save()
