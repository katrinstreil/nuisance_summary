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

# %%


def save():
    with open("data/7aP_P_draw_info.txt", "a") as myfile:
        myfile.write(
            str(float(bias_rnd[0]))
            + "    "
            + str(float(res_rnd[0]))
            + "    "
            + str(float(dataset.stat_sum()))
            + "\n"
        )
    with open("data/7aP_P_draw_par.txt", "a") as myfile:
        myfile.write(stri + "\n")
    with open("data/7aP_P_draw_flux.txt", "a") as myfile:
        myfile.write(ff + "\n")
    with open("data/7aP_P_draw_flux2e.txt", "a") as myfile:
        myfile.write(ff2 + "\n")

    with open("data/7aP_N_P_draw_par.txt", "a") as myfile:
        myfile.write(stri_N + "\n")
    with open("data/7aP_N_P_draw_flux.txt", "a") as myfile:
        myfile.write(ffN + "\n")
    with open("data/7aP_N_P_draw_flux2e.txt", "a") as myfile:
        myfile.write(ffN2 + "\n")


import sys

import astropy.units as u
import gammapy
import numpy as np
import pyximport
from gammapy.estimators import FluxPointsEstimator
from gammapy.maps import MapAxis
from gammapy.modeling import Fit, Parameter, Parameters
from gammapy.modeling.models import EffAreaIRFModel, ERecoIRFModel, IRFModels, Models
from gammapy.modeling.models.spectral import scale_plot_flux

sys.path.append("/home/katrin/Documents/nuisance_summary/")
sys.path.append("../../../")
import Dataset_load  # noqa: E402
from Dataset_Creation import sys_dataset  # noqa: E402

pyximport.install()

print(f"loaded gammapy version: {gammapy.__version__} ")
print("Supposed to be 1.0 (21-12-2022)")


scaled_amplitude = Parameter("amplitude", value=1e-12)
lambda_ = Parameter("lambda_", value=1 / 60)

dataset_asimov = Dataset_load.create_asimov(
    model="ecpl", source="PKSflare", parameters=Parameters([scaled_amplitude, lambda_])
)

mask = dataset_asimov.mask.data.sum(axis=2).sum(axis=1) > 0

ebins = dataset_asimov.counts.geom.axes[0].center[mask]
print(len(ebins))


shift = 0.0
tilt = 0.0
resolution = 0.0
bias = 0.1

N = 200

save_flux = True
save_fluxpoints = 0
save_fluxpoints_N = 0
dataset_N = True


for n in range(N):
    print()
    print("====" * 30)
    print(n)
    print("====" * 30)
    res_rnd = [0.0]
    # res_rnd = np.random.normal(0, resolution, 1)
    bias_rnd = np.random.normal(0, bias, 1)

    print(f"res {res_rnd}, bias {bias_rnd}")
    sys_d_cor = sys_dataset(
        dataset_asimov=dataset_asimov,
        shift=0,
        tilt=0,
        resolution=res_rnd[0],
        bias=bias_rnd,
        rnd=True,
    )
    dataset = sys_d_cor.create_dataset()

    fit_cor = Fit(store_trace=False)
    minuit_opts = {"tol": 0.001, "strategy": 2}
    fit_cor.optimize_opts = minuit_opts
    result_cor = fit_cor.run([dataset])

    stri = ""
    for p in ["amplitude", "index", "lambda_", "norm", "tilt"]:
        stri += (
            str(dataset.models.parameters[p].value)
            + "   "
            + str(dataset.models.parameters[p].error)
            + "   "
        )
    print(stri)

    fluxes = []
    for e in ebins:
        flux = dataset.models[0].spectral_model(e)
        fluxes.append(flux.value)

    ff = str()
    for f in fluxes:
        ff += str(f) + "  "
    # print(ff)

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
    ff2 = str()
    for f in fluxe2:
        ff2 += str(f) + "  "

    energy_edges = dataset.geoms["geom"].axes[0].edges[::2]

    dataset_N = sys_d_cor.create_dataset_N(e_reco_n=10, counts=dataset.counts)
    zero = 1e-24
    dataset_N.models = Models(
        [
            dataset_N.models[0],
            dataset_N.background_model,
            IRFModels(
                eff_area_model=None,
                e_reco_model=ERecoIRFModel(),
                datasets_names=dataset_N.name,
            ),
        ]
    )
    # addional parameter bias and resolution (ereco) but are frozen
    penalising_invcovmatrix = np.zeros((2, 2))
    # 'bias', 'resolution', 'norm_nuisance',  'tilt_nuisance',
    np.fill_diagonal(
        penalising_invcovmatrix,
        [1 / bias**2, 1 / zero**2],
    )
    dataset_N.penalising_invcovmatrix = penalising_invcovmatrix
    dataset_N.irf_model.e_reco_model.parameters["bias"].frozen = False
    dataset_N.irf_model.e_reco_model.parameters["resolution"].frozen = True

    fit_cor = Fit(store_trace=False)
    result_cor = fit_cor.run([dataset_N])

    stri_N = ""
    for p in ["amplitude", "index", "lambda_", "norm", "tilt", "bias", "resolution"]:
        stri_N += (
            str(dataset_N.models.parameters[p].value)
            + "   "
            + str(dataset_N.models.parameters[p].error)
            + "   "
        )
    print(stri_N)

    fluxes = []
    for e in ebins:
        flux = dataset_N.models[0].spectral_model(e)
        fluxes.append(flux.value)

    ffN = str()
    for f in fluxes:
        ffN += str(f) + "  "

    energy_bounds = (ebins[0], ebins[-1]) * u.TeV

    energy_min, energy_max = energy_bounds
    energy = MapAxis.from_energy_bounds(
        energy_min,
        energy_max,
        len(ebins),
    )

    fluxe2, _ = dataset_N.models[0].spectral_model._get_plot_flux(
        sed_type="dnde", energy=energy
    )
    fluxe2 = scale_plot_flux(fluxe2, energy_power=2)
    fluxe2 = fluxe2.quantity[:, 0, 0]
    fluxe2 = np.array(fluxe2)
    ffN2 = str()
    for f in fluxe2:
        ffN2 += str(f) + "  "

    if save_fluxpoints:
        esti = FluxPointsEstimator(energy_edges=energy_edges)
        fluxpoints = esti.run([dataset])
        esti = FluxPointsEstimator(energy_edges=energy_edges)
        fluxpoints_N = esti.run([dataset_N])
        fluxpoints_N.write(
            f"data/fluxpoints/6P_fluxpoints_N_{bias_rnd[0]:.6}_{res_rnd[0]:.6}.fits"
        )
        Models([dataset_N.models[0]]).write(
            f"data/fluxpoints/6P_model_N_{bias_rnd[0]:.6}_{res_rnd[0]:.6}.yaml"
        )
        fluxpoints.write(
            f"data/fluxpoints/6P_fluxpoints_{bias_rnd[0]:.6}_{res_rnd[0]:.6}.fits"
        )
        Models([dataset.models[0]]).write(
            f"data/fluxpoints/6P_model_{bias_rnd[0]:.6}_{res_rnd[0]:.6}.yaml"
        )

    save()
# %%
