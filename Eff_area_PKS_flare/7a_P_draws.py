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


N = 500
sigma_a = shift
sigma_i = tilt

save = True
save_flux = True
save_fluxpoints = True
save_fluxpoints_N = True
dataset_N = True


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

    if save:
        with open("data/7a_P_draw_info.txt", "a") as myfile:
            myfile.write(
                str(float(shift_rnd))
                + "    "
                + str(float(tilt_rnd))
                + "    "
                + str(float(dataset.stat_sum()))
                + "\n"
            )

    stri = ""
    for p in ["amplitude", "index", "norm", "tilt"]:
        stri += (
            str(dataset.models.parameters[p].value)
            + "   "
            + str(dataset.models.parameters[p].error)
            + "   "
        )
    print(stri)
    if save:
        with open("data/7a_P_draw_par.txt", "a") as myfile:
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
        with open("data/7a_P_draw_flux.txt", "a") as myfile:
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
        with open("data/7a_P_draw_flux2e.txt", "a") as myfile:
            myfile.write(ff + "\n")

    energy_edges = dataset.geoms["geom"].axes[0].edges[::2]
    if save_fluxpoints:
        esti = FluxPointsEstimator(energy_edges=energy_edges)
        fluxpoints = esti.run([dataset])
        fluxpoints.write(
            f"data/fluxpoints/6_fluxpoints_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits"
        )
        Models([dataset.models[0]]).write(
            f"data/fluxpoints/6_model_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits"
        )

    if dataset_N:
        dataset_N = sys_d_cor.create_dataset_N(e_reco_n=10)
        zero = 1e-24
        dataset_N.models = Models(
            [
                dataset_N.models[0],
                dataset_N.background_model,
                IRFModels(
                    eff_area_model=EffAreaIRFModel(), datasets_names=dataset_N.name
                ),
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

        fit_cor = Fit(store_trace=False)
        result_cor = fit_cor.run([dataset_N])

        stri = ""
        for p in [
            "amplitude",
            "index",
            "norm",
            "tilt",
            "norm_nuisance",
            "tilt_nuisance",
        ]:
            stri += (
                str(dataset_N.models.parameters[p].value)
                + "   "
                + str(dataset_N.models.parameters[p].error)
                + "   "
            )
        print(stri)
        if save:
            with open("data/7a_N_P_draw_par.txt", "a") as myfile:
                myfile.write(stri + "\n")

        fluxes = []
        for e in ebins:
            flux = dataset_N.models[0].spectral_model(e)
            fluxes.append(flux.value)

        ff = str()
        for f in fluxes:
            ff += str(f) + "  "
        # print(ff)
        if save:
            with open("data/7a_N_P_draw_flux.txt", "a") as myfile:
                myfile.write(ff + "\n")

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
        ff = str()
        for f in fluxe2:
            ff += str(f) + "  "
        if save:
            with open("data/7a_N_P_draw_flux2e.txt", "a") as myfile:
                myfile.write(ff + "\n")

        if save_fluxpoints_N:
            esti = FluxPointsEstimator(energy_edges=energy_edges)
            fluxpoints_N = esti.run([dataset_N])
            fluxpoints_N.write(
                f"data/fluxpoints/6_fluxpoints_N_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits"
            )
            Models([dataset_N.models[0]]).write(
                f"data/fluxpoints/6_model_N_{shift_rnd[0]:.6}_{tilt_rnd[0]:.6}.fits"
            )
