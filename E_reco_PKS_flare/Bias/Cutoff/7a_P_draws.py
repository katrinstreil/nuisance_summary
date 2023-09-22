import os
import sys

import astropy.units as u
import gammapy
import numpy as np
import pyximport
from gammapy.maps import MapAxis
from gammapy.modeling import Fit, Parameter, Parameters
from gammapy.modeling.models.spectral import scale_plot_flux

import Dataset_load
from Dataset_Creation import sys_dataset

pyximport.install()

sys.path.append("/home/katrin/Documents/nuisance_summary/")
sys.path.append("../")

print(f"loaded gammapy version: {gammapy.__version__} ")
print("Supposed to be 1.0 (21-12-2022)")

scaled_amplitude = Parameter("amplitude", value=1e-12)
cutoff = 60
cutoff_input_value = cutoff
lambda_ = Parameter("lambda_", value=1 / cutoff)
dataset_asimov = Dataset_load.create_asimov(
    model="ecpl", source="PKSflare", parameters=Parameters([scaled_amplitude, lambda_])
)


mask = dataset_asimov.mask.data.sum(axis=2).sum(axis=1) > 0

ebins = dataset_asimov.counts.geom.axes[0].center[mask]
print(len(ebins))

bias_assumed = 0.1
sigma_assumed = 0.1


N = 300

save = True
save_flux = True


path = os.getcwd() + "/Bias/Cutoff/"
print(path)


for n in range(N):
    print()
    print("====" * 30)
    print(n)
    print("====" * 30)

    bias_rnd = np.random.normal(0, bias_assumed, 1)
    sigma_rnd = np.random.normal(0, sigma_assumed, 1)
    print(f"bias:, {bias_rnd}, res: {sigma_rnd}")
    sys_d_cor = sys_dataset(
        dataset_asimov=dataset_asimov,
        shift=0,
        tilt=0,
        bias=bias_rnd,
        resolution=sigma_rnd,
        rnd=True,
    )
    dataset = sys_d_cor.create_dataset()
    fit_cor = Fit(store_trace=False)
    minuit_opts = {"tol": 0.001, "strategy": 2}
    fit_cor.optimize_opts = minuit_opts
    result_cor = fit_cor.run([dataset])

    if save:
        with open(path + "data/7a_P_draw_info.txt", "a") as myfile:
            myfile.write(
                str(float(bias_rnd))
                + "    "
                + str(float(sigma_rnd))
                + "    "
                + str(float(dataset.stat_sum()))
                + "\n"
            )

    stri = ""
    for p in ["lambda_", "index", "norm", "tilt"]:
        stri += (
            str(dataset.models.parameters[p].value)
            + "   "
            + str(dataset.models.parameters[p].error)
            + "   "
        )
    print(stri)
    if save:
        with open(path + "data/7a_P_draw_par.txt", "a") as myfile:
            myfile.write(stri + "\n")

    fluxes = []
    for e in ebins:
        flux = dataset.models[0].spectral_model(e)
        fluxes.append(flux.value)

    ff = str()
    for f in fluxes:
        ff += str(f) + "  "
    # print(ff)
    if save:
        with open(path + "data/7a_P_draw_flux.txt", "a") as myfile:
            myfile.write(ff + "\n")

    energy_bounds = (ebins[0], ebins[-1]) * u.TeV

    energy_min, energy_max = energy_bounds
    energy = MapAxis.from_energy_bounds(
        energy_min,
        energy_max,
        len(ebins),
    )

    fluxe2, _ = dataset.models[0].spectral_model._get_plot_flux(
        sed_type="dnde", energy=energy
    )
    fluxe2 = scale_plot_flux(fluxe2, energy_power=2)
    fluxe2 = fluxe2.quantity[:, 0, 0]
    fluxe2 = np.array(fluxe2)
    ff = str()
    for f in fluxe2:
        ff += str(f) + "  "
    if save:
        with open(path + "data/7a_P_draw_flux2e.txt", "a") as myfile:
            myfile.write(ff + "\n")

    energy_edges = dataset.geoms["geom"].axes[0].edges
    # esti  = FluxPointsEstimator(energy_edges= energy_edges)
    # fluxpoints = esti.run([dataset])
    # if save_flux:
    #    fluxpoints.write(f'data/fluxpoints/6_fluxpoints_{shift_rnd[0]:.6}.fits')
    # except:
    #    pass
