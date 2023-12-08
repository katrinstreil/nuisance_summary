def save():
    with open("data/7aP_P_draw_info.txt", "a") as myfile:
        myfile.write(
            str(float(shift_rnd))
            + "    "
            + str(float(tilt_rnd))
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


def make_writeable(cc):
    d = dict()
    for ii, seg in enumerate(cc.allsegs[0]):
        d["x"] = seg[:, 0].tolist()
        d["y"] = seg[:, 1].tolist()
    return d


def save_countour(c, name):
    d = make_writeable(c)
    with open(name, "w") as f:
        json.dump(d, f)


import json
import sys

import astropy.units as u
import gammapy
import numpy as np
import pyximport
from gammapy.estimators import FluxPointsEstimator
from gammapy.maps import MapAxis
from gammapy.modeling import Fit, Parameter, Parameters
from gammapy.modeling.models import EffAreaIRFModel, IRFModels, Models
from gammapy.modeling.models.spectral import scale_plot_flux

sys.path.append("/home/katrin/Documents/nuisance_summary/")
sys.path.append("../")
import Dataset_load  # noqa: E402
from Dataset_Creation import sys_dataset  # noqa: E402

pyximport.install()

print(f"loaded gammapy version: {gammapy.__version__} ")
print("Supposed to be 1.0 (21-12-2022)")


scaled_amplitude = Parameter("amplitude", value=1e-12)
dataset_asimov = Dataset_load.create_asimov(
    model="pl", source="PKSflare", parameters=Parameters([scaled_amplitude])
)

mask = dataset_asimov.mask.data.sum(axis=2).sum(axis=1) > 0

ebins = dataset_asimov.counts.geom.axes[0].center[mask]
print(len(ebins))


shift = +0.1
tilt = 0.02


N = 100
sigma_a = shift
sigma_i = tilt

save_flux = True
dataset_N = True
advanced = True
compute_fluxpoints = True

for n in range(N):
    print()
    print("====" * 30)
    print(n)
    print("====" * 30)

    shift_rnd = np.random.normal(0, shift, 1)
    tilt_rnd = np.random.normal(0, tilt, 1)
    print(f"shift:, {shift_rnd}, tilt: {tilt_rnd}")
    sys_d_cor = sys_dataset(
        dataset_asimov=dataset_asimov, shift=shift_rnd, tilt=tilt_rnd, rnd=True
    )
    dataset = sys_d_cor.create_dataset()
    fit_cor = Fit(store_trace=False)
    minuit_opts = {"tol": 0.001, "strategy": 2}
    fit_cor.optimize_opts = minuit_opts
    result_cor = fit_cor.run([dataset])

    stri = ""
    for p in ["amplitude", "index", "norm", "tilt"]:
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
            IRFModels(eff_area_model=EffAreaIRFModel(), datasets_names=dataset_N.name),
        ]
    )
    # addional parameter bias and resolution (ereco) but are frozen
    penalising_invcovmatrix = np.zeros((2, 2))
    # 'bias', 'resolution', 'norm_nuisance',  'tilt_nuisance',
    np.fill_diagonal(
        penalising_invcovmatrix,
        [1 / shift**2, 1 / tilt**2],
    )
    dataset_N.penalising_invcovmatrix = penalising_invcovmatrix
    dataset_N.irf_model.eff_area_model.parameters["norm_nuisance"].frozen = False
    dataset_N.irf_model.eff_area_model.parameters["tilt_nuisance"].frozen = False

    fit_cor_N = Fit(store_trace=False)
    result_cor_N = fit_cor_N.run([dataset_N])

    stri_N = ""
    for p in ["amplitude", "index", "norm", "tilt", "norm_nuisance", "tilt_nuisance"]:
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

    print("fit succes:", result_cor.success, " Nui: ", result_cor_N.success)
    if advanced and result_cor.success and result_cor_N.success:
        # if compute_fluxpoints:
        models = Models([dataset.models[0].copy()])
        models_N = Models([dataset_N.models[0].copy()])

        print("Computing contour")
        c = fit_cor.minuit.draw_mncontour(
            "par_000_index",
            "par_001_amplitude",
            cl=0.68,
            size=30,
        )
        c_N = fit_cor_N.minuit.draw_mncontour(
            "par_000_index",
            "par_001_amplitude",
            cl=0.68,
            size=30,
        )
        print("Computing contour 39 %")

        c_39 = fit_cor.minuit.draw_mncontour(
            "par_000_index",
            "par_001_amplitude",
            cl=0.39,
            size=30,
        )
        c_N_39 = fit_cor_N.minuit.draw_mncontour(
            "par_000_index",
            "par_001_amplitude",
            cl=0.39,
            size=30,
        )

        esti = FluxPointsEstimator(energy_edges=energy_edges)
        fluxpoints = esti.run([dataset])
        esti = FluxPointsEstimator(energy_edges=energy_edges)
        fluxpoints_N = esti.run([dataset_N])

        fluxpoints_N.write(
            f"data/fluxpoints/6P_fluxpoints_N_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits"
        )
        models_N.write(
            f"data/fluxpoints/6P_model_N_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.yaml"
        )
        fluxpoints.write(
            f"data/fluxpoints/6P_fluxpoints_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits"
        )
        models.write(
            f"data/fluxpoints/6P_model_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.yaml"
        )

        save_countour(
            c, f"data/contours/6P_068_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.json"
        )
        save_countour(
            c_N, f"data/contours/6P_068_N_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.json"
        )
        save_countour(
            c_39, f"data/contours/6P_039_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.json"
        )
        save_countour(
            c_N_39, f"data/contours/6P_039_N_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.json"
        )

    if result_cor.success and result_cor_N.success:
        save()
