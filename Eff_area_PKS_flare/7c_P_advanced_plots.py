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

import json

# %%
import sys

import astropy.units as u
import gammapy
import matplotlib.pylab as plt
import numpy as np
import pyximport
import seaborn as sns
import yaml
from gammapy.datasets import FluxPointsDataset
from gammapy.estimators import FluxPoints
from gammapy.estimators.map.core import DEFAULT_UNIT, FluxMaps
from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import FoVBackgroundModel, Models
from gammapy.modeling.models.spectral import scale_plot_flux
from matplotlib import rc
from scipy.interpolate import interp2d

# import random
sys.path.append("/home/katrin/Documents/nuisance_summary/")
sys.path.append("../")
import Dataset_load  # noqa: E402
from Dataset_Creation import sys_dataset  # noqa: E402

print(f"loaded gammapy version: {gammapy.__version__} ")
print("Supposed to be 1.0 (21-12-2022)")
rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
pyximport.install()

# %%
savefig = 1
picformat = "png"
fig, a_fake = plt.subplots(1, 1, figsize=(5.0, 5.0))


# %% [markdown]
# ## Methods
#


# %%
def compute_fraction(Ls_new, x_new, y_new, threshold, plot=False):
    tot_int = np.sum(Ls_new)
    offset = Ls_new.min() + threshold
    inside = Ls_new * (Ls_new >= offset)
    fraction_inside = np.sum(inside) / tot_int
    if plot:
        fig, ax = plt.subplots(
            1,
            1,
        )
        im = ax.pcolormesh(x_new, y_new, Ls_new)
        fig.colorbar(im)
        ax.set_title(f"frac inside : {fraction_inside * 100:.4} %")

        ax.contour(
            x_new, y_new, Ls_new, colors=("gold",), levels=[Ls_new.min() + threshold]
        )
    return fraction_inside


def plot_L(
    Ls_new,
    x_new,
    y_new,
    threshold,
    find_min,
    color="red",
    ax=None,
    fig=None,
    cbar=True,
    cmap="viridis",
):
    index_min, index_max, ampli_min, ampli_max, index_best, ampli_best = compute_errors(
        Ls_new, x_new, y_new, threshold, find_min=find_min
    )
    if ax is None:
        fig, ax = plt.subplots(
            1,
            1,
        )

    im = ax.pcolormesh(x_new, y_new, Ls_new, cmap=cmap, rasterized=True)
    if cbar:
        fig.colorbar(im)

    CS = ax.contour(
        x_new, y_new, Ls_new, colors=(color,), levels=[Ls_new.min() + threshold]
    )
    # ax.clabel(CS, CS.levels, inline=True, fmt='$1\\sigma$', fontsize=12)

    plot_errors(
        ax,
        x_new,
        y_new,
        index_min,
        index_max,
        ampli_min,
        ampli_max,
        index_best,
        ampli_best,
        color,
    )

    ax.plot(ampli_best, index_best, "x", color=color)

    ax.set_ylabel(f"Amplitude [{a_unit}]")
    ax.set_xlabel("Index []")
    return CS


def plot_errors(
    ax,
    x_new,
    y_new,
    index_min,
    index_max,
    ampli_min,
    ampli_max,
    index_best,
    ampli_best,
    color,
):
    linewidth = 2

    ax.hlines(
        y=index_min,
        xmin=x_new[0],
        xmax=x_new[-1],
        color=color,
        linestyle="dotted",
        linewidth=linewidth,
    )
    ax.hlines(
        y=index_max,
        xmin=x_new[0],
        xmax=x_new[-1],
        color=color,
        linestyle="dotted",
        linewidth=linewidth,
    )

    ax.vlines(
        x=ampli_min,
        ymin=y_new[0],
        ymax=y_new[-1],
        color=color,
        linestyle="dotted",
        linewidth=linewidth,
    )
    ax.vlines(
        x=ampli_max,
        ymin=y_new[0],
        ymax=y_new[-1],
        color=color,
        linestyle="dotted",
        linewidth=linewidth,
    )


def compute_errors(Ls_new, x_new, y_new, threshold, find_min):
    offset = Ls_new.min() + threshold
    if offset > 0:
        inside = Ls_new * (Ls_new < offset)
    else:
        inside = Ls_new * (Ls_new >= offset)
    if find_min:  # find min:
        pos = np.where(inside > 0)
        ampli_best = x_new[np.where(Ls_new == Ls_new.min())[1][0]]
        index_best = y_new[np.where(Ls_new == Ls_new.min())[0][0]]

    else:  # find max
        pos = np.where(inside > 0)
        ampli_best = x_new[np.where(Ls_new == Ls_new.max())[1][0]]
        index_best = y_new[np.where(Ls_new == Ls_new.max())[0][0]]

    delta_bin = (x_new[1] - x_new[0]) / 2
    ampli_min = x_new[np.min(pos[1])] - delta_bin
    ampli_max = x_new[np.max(pos[1])] + delta_bin
    ampli_best += delta_bin

    delta_bin = (y_new[1] - y_new[0]) / 2
    index_min = y_new[np.min(pos[0])] - delta_bin
    index_max = y_new[np.max(pos[0])] + delta_bin
    index_best += delta_bin

    return index_min, index_max, ampli_min, ampli_max, index_best, ampli_best


def fraction_within(lower_lim, upper_lim, data):
    is_within = []
    for d in data:
        larger = np.all(d[0] >= lower_lim)
        smaller = np.all(d[0] <= upper_lim)
        is_within.append(larger and smaller)

    return np.count_nonzero(is_within) / len(data)


# %%
def plot_amplitude(axs_a):
    # ampli
    a = axs_a
    i = 1
    a.hist(
        valuies[i][1:, 0],
        color=color_rnd,
        alpha=0.8,
        bins=np.linspace(xlims[i][0], xlims[i][-1], nbins),
        orientation="horizontal",
    )

    a.hist(
        valuies_N[i][1:, 0],
        color=color_rnd_nui,
        alpha=0.8,
        bins=np.linspace(xlims[i][0], xlims[i][-1], nbins),
        orientation="horizontal",
    )

    ylim = a.get_xlim()
    delta_ylim = 5
    a.hlines(
        valuies_asimov_N[i][0],
        ylim[0],
        ylim[1] + delta_ylim,
        color=colors[1],
        linestyle="dashed",
    )
    a.fill_betweenx(
        [
            valuies_asimov_N[i][0] - valuies_asimov_N[i][1],
            valuies_asimov_N[i][0] + valuies_asimov_N[i][1],
        ],
        ylim[0],
        ylim[1] + delta_ylim,
        alpha=alpha_nui,
        color=colors[3],
        label=legends[0] + f"{withins[i][1]*100 :.3}\\%",
    )
    a.hlines(valuies_asimov[i][0], ylim[0], ylim[1], color=colors[0])
    a.fill_betweenx(
        [
            valuies_asimov[i][0] - valuies_asimov[i][1],
            valuies_asimov[i][0] + valuies_asimov[i][1],
        ],
        ylim[0],
        ylim[1],
        alpha=alpha_st,
        color=colors[2],
        label=legends[1] + f"{withins[i][0]*100 :.3}\\%",
    )
    a.set_xlabel("")
    a.set_ylim(axs_L.get_ylim()[0], axs_L.get_ylim()[1])
    yy = a.get_xlim()[1]
    print(
        f"amplitudes  : {np.mean(amplitudes[:,0]):.5} pm {np.std(amplitudes[:,0]):.5}"
    )
    print(
        f"amplitudes N: {np.mean(amplitudes_N[:,0]):.5} pm {np.std(amplitudes_N[:,0]):.5}"
    )

    a.errorbar(
        x=yy / 3,
        y=np.mean(amplitudes[:, 0]),
        yerr=np.std(amplitudes[:, 0]),
        fmt="o",
        color=color_rnd_err,
        label="",
    )

    a.errorbar(
        x=yy / 3 - y2,
        y=np.mean(amplitudes_N[:, 0]),
        yerr=np.std(amplitudes_N[:, 0]),
        fmt="o",
        color=color_rnd_nui_err,
        label="",
    )

    a.set_yticklabels([])
    a.set_yticks([])


def plot_index(axs_i):
    # index
    a = axs_i
    i = 2
    a.hist(
        valuies[i][1:, 0],
        color=color_rnd,
        alpha=0.8,
        bins=np.linspace(xlims[i][0], xlims[i][-1], nbins),
    )
    a.hist(
        valuies_N[i][1:, 0],
        color=color_rnd_nui,
        alpha=0.8,
        bins=np.linspace(xlims[i][0], xlims[i][-1], nbins),
    )
    ylim = a.get_ylim()
    delta_ylim = 5
    a.vlines(
        valuies_asimov_N[i][0],
        ylim[0],
        ylim[1] + delta_ylim,
        color=colors[1],
        linestyle="dashed",
    )
    a.fill_between(
        [
            valuies_asimov_N[i][0] - valuies_asimov_N[i][1],
            valuies_asimov_N[i][0] + valuies_asimov_N[i][1],
        ],
        ylim[0],
        ylim[1] + delta_ylim,
        alpha=alpha_nui,
        color=colors[3],
        label=legends[0] + f"{withins[i][1]*100 :.3}\\%",
    )
    a.vlines(valuies_asimov[i][0], ylim[0], ylim[1], color=colors[0])
    a.fill_between(
        [
            valuies_asimov[i][0] - valuies_asimov[i][1],
            valuies_asimov[i][0] + valuies_asimov[i][1],
        ],
        ylim[0],
        ylim[1],
        alpha=alpha_st,
        color=colors[2],
        label=legends[1] + f"{withins[i][0]*100 :.3}\\%",
    )
    a.set_xlabel(xlabels[i])
    yy = a.get_ylim()[1]
    print(f"index     : {np.mean(indices[:,0]):.5} pm {np.std(indices[:,0]):.5}")
    print(f"index nui : {np.mean(indices_N[:,0]):.5} pm {np.std(indices_N[:,0]):.5}")
    a.errorbar(
        x=np.mean(indices[:, 0]),
        y=yy / 3,
        xerr=np.std(indices[:, 0]),
        fmt="o",
        color=color_rnd_err,
        label="",
    )
    a.errorbar(
        x=np.mean(indices_N[:, 0]),
        y=yy / 3 - y2,
        xerr=np.std(indices_N[:, 0]),
        fmt="o",
        color=color_rnd_nui_err,
        label="",
    )
    a.set_xlim(axs_L.get_xlim()[0], axs_L.get_xlim()[1])
    a.set_xlabel("")
    a.set_xticklabels([])
    a.set_xticks([])


def plot_randoms(axs_L, fig):
    (
        ampli_min_asimov_N,
        ampli_max_asimov_N,
        index_min_asimov_N,
        index_max_asimov_N,
        ampli_best_asimov_N,
        index_best_asimov_N,
    ) = compute_errors(
        data_contour_N, indexy__new_N, amplix__new_N, threshold_contour, find_min=True
    )

    CS_N = plot_L(
        data_contour_N,
        indexy__new_N,
        amplix__new_N,
        threshold_contour,
        find_min=True,
        color=colors[1],
        ax=a_fake,
        fig=fig,
        cbar=False,
        cmap=cmap,
    )
    dat_N = CS_N.allsegs[0][0]
    axs_L.plot(dat_N[:, 0], dat_N[:, 1], color=colors[1])

    CS = plot_L(
        data_contour,
        indexy__new,
        amplix__new,
        threshold_contour,
        find_min=True,
        color=colors[1],
        ax=a_fake,
        fig=fig,
        cbar=False,
        cmap=cmap,
    )
    dat = CS.allsegs[0][0]
    axs_L.plot(dat[:, 0], dat[:, 1], color=colors[0])
    # axs_L.plot(index_best_asimov, ampli_best_asimov, "x", color=colors[0])

    axs_L.scatter(
        indices_data_N,
        amplitudes_data_N,
        marker=".",
        color=color_rnd_nui,
        alpha=alpha_rnd_nui,
        label="Rnd with nui.",
    )
    axs_L.scatter(
        indices_data,
        amplitudes_data,
        marker="x",
        s=14,
        color=color_rnd,
        alpha=alpha_rnd,
        label="Rnd w/o nui.",
    )


def plot_best_fit_errobar(axs_L, model_N, model):  # = fluxpoints_ds.models[0], )
    axs_L.plot(2.3, 1e-12, "^", markersize=6, color="black", label="Input")
    axs_L.plot(2.3, 1e-12, "x", markersize=8, color=colors[1], label=legendsscatter[1])
    axs_L.plot(2.3, 1e-12, "+", markersize=10, color=colors[0], label=legendsscatter[0])

    off = 1e-14
    axs_L.text(
        index_best_asimov_N,
        off + ampli_max_asimov_N,
        "$1\\sigma$",
        color=colors[1],
        fontsize="x-large",
    )
    axs_L.text(
        index_best_asimov_N,
        off + ampli_max_asimov,
        "$1\\sigma$",
        color=colors[0],
        fontsize="x-large",
    )
    axs_L.plot(
        [np.nan, np.nan],
        [np.nan, np.nan],
        alpha=1,
        color=colors[0],
        label=legends[0],
    )  # + f'{fraction_N:.3}\\%')
    axs_L.plot(
        [np.nan, np.nan],
        [np.nan, np.nan],
        alpha=1,
        color=colors[1],
        label=legends[1],
    )  # +  f'{fraction:.3}\\%'  )


# def plot_best_fit_errobar(axs_L, model_N = fluxpoints_Nds.models[0], model = fluxpoints_ds.models[0], colors= [color_rnd_nui_example,color_rnd_example] ):
def plot_best_fit_errobar(
    axs_L, model_N, model, colors, label, alpha=[1.0, 1.0]
):  # = [color_rnd_nui_example,color_rnd_example] ):
    ampli, index = (
        model_N.parameters["amplitude"].value,
        model_N.parameters["index"].value,
    )
    ampli_err, index_err = (
        model_N.parameters["amplitude"].error,
        model_N.parameters["index"].error,
    )

    axs_L.errorbar(
        index,
        ampli,
        xerr=index_err,
        yerr=ampli_err,
        marker="o",
        markersize=6,
        linewidth=2,
        color=colors[0],
        alpha=alpha[0],
        label=label[0],
        capsize=5,
    )

    ampli, index = model.parameters["amplitude"].value, model.parameters["index"].value
    ampli_err, index_err = (
        model.parameters["amplitude"].error,
        model.parameters["index"].error,
    )
    axs_L.errorbar(
        index,
        ampli,
        xerr=index_err,
        yerr=ampli_err,
        marker="x",
        markersize=6,
        color=colors[1],
        linewidth=2,
        alpha=alpha[1],
        label=label[1],
        capsize=5,
    )


def plot_fp(fp, ax=None, sed_type="dnde", energy_power=2, xerr=False, **kwargs):
    """Plot flux points.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axis object to plot on.
    sed_type : {"dnde", "flux", "eflux", "e2dnde"}
        Sed type
    energy_power : float
        Power of energy to multiply flux axis with
    **kwargs : dict
        Keyword arguments passed to `~RegionNDMap.plot`

    Returns
    -------
    ax : `~matplotlib.axes.Axes`
        Axis object
    """
    if sed_type is None:
        sed_type = fp.sed_type_plot_default

    if not fp.norm.geom.is_region:
        raise ValueError("Plotting only supported for region based flux points")

    if ax is None:
        ax = plt.gca()

    flux_unit = DEFAULT_UNIT[sed_type]

    flux = getattr(fp, sed_type)

    # get errors and ul
    y_errn, y_errp = fp._plot_get_flux_err(sed_type=sed_type)
    x_errp = 0
    is_ul = fp.is_ul.data

    if fp.has_ul and y_errn and is_ul.any():
        flux_ul = getattr(fp, sed_type + "_ul").quantity
        y_errn.data[is_ul] = np.clip(
            0.5 * flux_ul[is_ul].to_value(y_errn.unit), 0, np.inf
        )
        y_errp.data[is_ul] = 0
        flux.data[is_ul] = flux_ul[is_ul].to_value(flux.unit)
        kwargs.setdefault("uplims", is_ul)

    # set flux points plotting defaults
    if y_errp and y_errn:
        y_errp = np.clip(
            scale_plot_flux(y_errp, energy_power=energy_power).quantity, 0, np.inf
        )
        y_errn = np.clip(
            scale_plot_flux(y_errn, energy_power=energy_power).quantity, 0, np.inf
        )
        kwargs.setdefault("yerr", (y_errn, y_errp))
    else:
        kwargs.setdefault("yerr", None)

    if xerr == False:
        kwargs.setdefault("xerr", None)

    flux = scale_plot_flux(flux=flux.to_unit(flux_unit), energy_power=energy_power)
    ax = flux.plot(ax=ax, **kwargs)
    ax.set_ylabel(f"{sed_type} [{ax.yaxis.units}]")
    ax.set_yscale("log")
    return ax


def plot_choice_sed(fig, ax, fluxpoints_Nds):
    plot_fp(
        fluxpoints_Nds.data,
        ax=ax,
        color=color_rnd_nui_example,
        label="",
        marker="x",
        capsize=5,
    )
    plot_fp(
        fluxpoints_Nds.data,
        xerr=True,
        ax=ax,
        color=color_rnd_nui_example,
        label="SED with nui.",
        marker="x",
    )
    plot_fp(
        fluxpoints_ds.data,
        energy_power=2,
        ax=ax,
        color=color_rnd_example,
        label="",
        marker="x",
        capsize=5,
    )
    plot_fp(
        fluxpoints_ds.data,
        energy_power=2,
        ax=ax,
        xerr=True,
        color=color_rnd_example,
        label="SED w/0 nui.",
        marker="x",
    )

    model = fluxpoints_ds.models[0]
    info = f"$\\phi_0$={model.parameters['amplitude'].value:.2} $\\pm$"
    info += f"{model.parameters['amplitude'].error:.2}\n  {model.parameters['amplitude'].unit}"
    info += f"\n$\\Gamma$={model.parameters['index'].value:.2} $\\pm$"
    info += f"{model.parameters['index'].error:.2}  {model.parameters['index'].unit}"
    fluxpoints_ds.models[0].spectral_model.plot(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        color=color_rnd_example,
        label="Best-fit spectrum w/0 nui.",
    )
    fluxpoints_ds.models[0].spectral_model.plot_error(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        facecolor=color_rnd_example,
        label="",
        alpha=alpha_example,
    )

    model = fluxpoints_Nds.models[0]
    info = f"$\\phi_0$={model.parameters['amplitude'].value:.2} $\\pm$"
    info += f"{model.parameters['amplitude'].error:.2}\n  {model.parameters['amplitude'].unit}"
    info += f"\n$\\Gamma$={model.parameters['index'].value:.2} $\\pm$"
    info += f"{model.parameters['index'].error:.2}  {model.parameters['index'].unit}"
    fluxpoints_Nds.models[0].spectral_model.plot(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        color=color_rnd_nui_example,
        label="Best-fit spectrum with nui.",
        linestyle="dashed",
    )
    fluxpoints_Nds.models[0].spectral_model.plot_error(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        facecolor=color_rnd_nui_err,
        label="",
        alpha=alpha_nui_example,
    )

    fluxpoints_ds.models[0].spectral_model.plot_error(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        facecolor=color_rnd_err,
        label="",
        alpha=alpha_example,
    )

    ax.set_title(
        f"Eff. Area Sys Shift: {shift_rnd*100:.3} \\%; Tilt: {tilt_rnd*100:.3}\\% "
    )
    model = dataset_asimov.models[0]
    info = f"$\\phi_0$={model.parameters['amplitude'].value:.2}\n"
    info += f"{model.parameters['amplitude'].unit}"
    info += f"\n$\\Gamma$={model.parameters['index'].value:.2} "
    info += f"{model.parameters['index'].unit}"
    dataset_asimov.models[0].spectral_model.plot(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        color="black",
        label="Input",
        linestyle="dotted",
    )
    # ax.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.6), ncol = 3)
    ax.legend(loc="lower left")

    # ax.legend()
    # plt.tight_layout()
    ax.set_xlim(0.3, 100)
    ax.set_ylim(1e-13, 2e-12)


# %%
def plot_rnd_fistribution(fig, ax, cc):
    nbins = 30
    count, bins, ignored = axs.hist(shift_rnds, nbins, density=True, color="grey")
    sigma = shift
    mu = 0
    axs.plot(
        bins,
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((bins - mu) ** 2) / (2 * sigma**2)),
        linewidth=2,
        color="black",
        label=f"$\\mu = {mu}; \\sigma = {sigma}$",
    )

    axs.vlines(
        shift_rnd,
        0,
        axs.get_ylim()[1],
        color=cc,
        label=f"Example value: {shift_rnd*100:.2} \\%",
    )

    axs.set_xlabel("Systematic on Eff. Area Norm [\\%]")
    axs.legend(loc="center right")

    x = axs.get_xlim()
    deltax = np.max([mu - x[0], x[1] - mu])
    axs.set_xlim(mu - deltax, mu + deltax)

    count, bins, ignored = ax1.hist(index_rnds, nbins, density=True, color="grey")
    sigma = tilt
    mu = 0
    ax1.plot(
        bins,
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((bins - mu) ** 2) / (2 * sigma**2)),
        linewidth=2,
        color="black",
        label=f"$\\mu = {mu}; \\sigma = {sigma:.3}$",
    )
    ax1.vlines(
        tilt_rnd,
        0,
        ax1.get_ylim()[1],
        color=cc,
        label=f"Example value {tilt_rnd*100:.2} \\%",
    )

    ax1.set_xlabel("Systematic on Eff. Area Tilt [\\%]")
    # ax1.set_title(f"N = {N}")
    ax1.legend(loc="center right")

    x = ax1.get_xlim()
    deltax = np.max([mu - x[0], x[1] - mu])
    ax1.set_xlim(mu - deltax, mu + deltax)
    plt.tight_layout()


def plot_asimov_spectrum(fig, ax):
    model = dataset_asimov_N.models[0].spectral_model
    model.plot(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        color=colors[1],
        label="Asimov Fit with nui.",
        linestyle="solid",
    )

    model.plot_error(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        facecolor=colors[3],
        label="",
        alpha=1,
    )

    model = dataset_asimov.models[0].spectral_model
    model.plot(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        color=colors[0],
        linestyle="dashed",
        label="Asimov Fit w/0 nui.",
    )
    model.plot_error(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        facecolor=colors[2],
        label="",
        alpha=1,
    )

    model = dataset_asimov.models[0]
    dataset_asimov.models[0].spectral_model.plot(
        energy_bounds=[0.3, 100] * u.TeV,
        energy_power=2,
        ax=ax,
        color="black",
        label="Input",
        linestyle="dotted",
    )

    ax.legend(loc="lower left")
    ax.set_xlim(0.3, 100)
    ax.set_ylim(1e-13, 2e-12)


def plot_coverage(fig, ax2):
    p = compute_precision(N)
    ax2.errorbar(ebins, withine, yerr=p, color=colors[0], label="")
    ax2.errorbar(ebins * 1.05, withine_N, yerr=p, color=colors[1], label="")
    ax2.plot(ebins, [0.68] * len(ebins), ":", color="black", label="")
    # ax2.legend(ncol = 2)
    axs.set_title(f"N = {N} (p = {p*100:.3}\\%)")
    ax2.set_xlabel("Energy [TeV]")
    ax2.set_ylabel("Coverage")
    axs.set_xlabel("")
    plt.tight_layout()


# %% [markdown]
# ## Import

# %% [markdown]
# ### Datasets

# %%
scaled_amplitude = Parameter("amplitude", value=1e-12)
dataset_asimov = Dataset_load.create_asimov(
    model="pl", source="PKSflare", parameters=Parameters([scaled_amplitude])
)

shift = 0.1
tilt = 0.02

sys_ = sys_dataset(
    dataset_asimov=dataset_asimov,
    shift=0,
    tilt=0,
    bias=0,
    resolution=0,
    rnd=False,
    e_reco_creation=10,
)
dataset_asimov = sys_.create_dataset()
dataset_asimov_N = sys_.create_dataset_N(e_reco_n=10)
zero = 1e-24
# addional parameter bias and resolution (ereco) but are frozen
penalising_invcovmatrix = np.zeros((4, 4))
# 'bias', 'resolution', 'norm_nuisance',  'tilt_nuisance',
np.fill_diagonal(
    penalising_invcovmatrix,
    [1 / zero**2, 1 / zero**2, 1 / shift**2, 1 / tilt**2],
)
dataset_asimov_N.penalising_invcovmatrix = penalising_invcovmatrix
dataset_asimov_N.irf_model.e_reco_model.parameters.freeze_all()
dataset_asimov_N.irf_model.parameters["tilt_nuisance"].frozen = False

a_unit = dataset_asimov_N.models.parameters["amplitude"].unit

# %%
m = Models.read("data/1_model_cor.yml")
dataset_asimov.models = Models(
    [
        m[0],
        FoVBackgroundModel(dataset_name=dataset_asimov.name),
    ]
)
path = f"data/2_model_nui_{0.0}_tilt_{0}.yml"
dataset_asimov_N = Dataset_load.load_dataset_N(dataset_asimov_N, path)
dataset_asimov_N.models.parameters["tilt_nuisance"].frozen = False

# %% [markdown]
# ### Fluxpoints

# %%
fp_asimov = FluxPointsDataset(
    data=FluxPoints.read("data/fluxpoints/6_fluxpoints_asimov.fits"),
    models=Models.read("data/fluxpoints/6_model_asimov.fits"),
)
fp_asimov_N = FluxPointsDataset(
    data=FluxPoints.read("data/fluxpoints/6_fluxpoints_asimov_N.fits"),
    models=Models.read("data/fluxpoints/6_model_asimov_N.fits"),
)


# %% [markdown]
# ### Results

# %%
infos = np.loadtxt("data/7aP_P_draw_info.txt")
pars = np.loadtxt("data/7aP_P_draw_par.txt")
flux = np.loadtxt("data/7aP_P_draw_flux.txt")
flux2e = np.loadtxt("data/7aP_P_draw_flux2e.txt")


lem = len(infos)
N_max = 1000
N = np.min([lem, N_max])

print(N)
flux = flux[:N, :]
shift_rnds = infos[:N_max, 0]
index_rnds = infos[:N_max, 1]
stat_sums = infos[:N_max, 2]


# amplitude', 'index', 'norm', 'tilt
amplitudes = pars[:N_max, 0:2]
indices = pars[:N_max, 2:4]
norms = pars[:N_max, 4:6]
tilts = pars[:N_max, 6:8]

valuies = [norms, amplitudes, indices]


# %%
pars_N = np.loadtxt("data/7aP_N_P_draw_par.txt")
flux_N = np.loadtxt("data/7aP_N_P_draw_flux.txt")
flux2e_N = np.loadtxt("data/7aP_N_P_draw_flux2e.txt")


lem = len(pars_N)
N_max = 1000
N = np.min([lem, N_max])

print(N)

shift_rnds = infos[:N_max, 0]
index_rnds = infos[:N_max, 1]
stat_sums = infos[:N_max, 2]


# amplitude', 'index', 'norm', 'tilt
amplitudes_N = pars_N[:N_max, 0:2]
indices_N = pars_N[:N_max, 2:4]
norms_N = pars_N[:N_max, 4:6]
tilts_N = pars_N[:N_max, 6:8]
norm_nuisance_N = pars_N[:N_max, 8:10]
tilt_nuisance_N = pars_N[:N_max, 10:12]

valuies_N = [norms_N, amplitudes_N, indices_N, norm_nuisance_N, tilt_nuisance_N]


amplitudes_data_N = valuies_N[1][:, 0]
indices_data_N = valuies_N[2][:, 0]

# %%
fig, (axs, ax1) = plt.subplots(1, 2, figsize=(6, 3))
nbins = 30
count, bins, ignored = axs.hist(
    norm_nuisance_N[:, 0], nbins, density=True, color="grey"
)

axs.set_xlabel("Best Fit Systematic on Eff. Area Norm [\\%]")
axs.legend(loc="center right")

x = axs.get_xlim()
mu = 0
axs.set_xlim(mu - np.max(np.abs(x)), mu + np.max(np.abs(x)))

count, bins, ignored = ax1.hist(
    tilt_nuisance_N[:, 0], nbins, density=True, color="grey"
)

ax1.set_xlabel("Best Fit Systematic on Eff. Area Tilt [\\%]")
# ax1.set_title(f"N = {N}")
ax1.legend(loc="center right")

x = ax1.get_xlim()
ax1.set_xlim(mu - np.max(np.abs(x)), mu + np.max(np.abs(x)))
plt.tight_layout()

print(
    f"Norm nuisance error: {np.mean(norm_nuisance_N[:,1]):.3} pm {np.std(norm_nuisance_N[:,1]):.3}"
)
print(
    f"Tilt nuisance error: {np.mean(tilt_nuisance_N[:,1]):.3} pm {np.std(tilt_nuisance_N[:,1]):.3}"
)

# %%
valuies_asimov = [
    [
        1,  # dataset_asimov.models[1].parameters['norm'].value,
        0,
    ],  # dataset_asimov.models[1].parameters['norm'].error],
    [
        dataset_asimov.models[0].parameters["amplitude"].value,
        dataset_asimov.models[0].parameters["amplitude"].error,
    ],
    [
        dataset_asimov.models[0].parameters["index"].value,
        dataset_asimov.models[0].parameters["index"].error,
    ],
]


valuies_asimov_N = [
    [
        dataset_asimov_N.models.parameters["norm"].value,
        dataset_asimov_N.models.parameters["norm"].error,
    ],
    [
        dataset_asimov_N.models.parameters["amplitude"].value,
        dataset_asimov_N.models.parameters["amplitude"].error,
    ],
    [
        dataset_asimov_N.models.parameters["index"].value,
        dataset_asimov_N.models.parameters["index"].error,
    ],
]
print(valuies_asimov_N)

# %% [markdown]
# # Likelihood Contours

# %% [markdown]
# ### Without Nuisance Parameters

# %%
with open("data/7c_contour.yml", "r") as stream:
    contour = yaml.safe_load(stream)
source = list(contour.keys())[0][:-24]
amplix__ = contour[f"{source}.spectral.amplitude_scan"]
indexy__ = contour[f"{source}.spectral.index_scan"]
N_new = 110
N_new_y = 100
amplix__new = np.linspace(amplix__[0], amplix__[-1], N_new)
indexy__new = np.linspace(indexy__[0], indexy__[-1], N_new_y)


f = interp2d(
    x=indexy__,
    y=amplix__,
    z=contour["stat_scan"],
    kind="cubic",
    fill_value=None,
    bounds_error=False,
)
data_contour = f(indexy__new, amplix__new)


# %%
threshold_contour = 1

(
    ampli_min_asimov,
    ampli_max_asimov,
    index_min_asimov,
    index_max_asimov,
    ampli_best_asimov,
    index_best_asimov,
) = compute_errors(
    data_contour, indexy__new, amplix__new, threshold_contour, find_min=True
)
CS = plot_L(
    data_contour,
    indexy__new,
    amplix__new,
    threshold_contour,
    find_min=True,
    color="lightblue",
)

dat = CS.allsegs[0][0]

# %% [markdown]
# ### With Nuisance Parameters

# %%

with open("data/7c_contour_N.yml", "r") as stream:
    contour_N = yaml.safe_load(stream)

# print(contour_N)
source_N = list(contour_N.keys())[0][:-24]

amplix___N = contour_N[f"{source_N}.spectral.amplitude_scan"]
indexy___N = contour_N[f"{source_N}.spectral.index_scan"]
amplix__new_N = np.linspace(amplix___N[0], amplix___N[-1], N_new)
indexy__new_N = np.linspace(indexy___N[0], indexy___N[-1], N_new_y)


f = interp2d(
    x=indexy___N,
    y=amplix___N,
    z=contour_N["stat_scan"],
    kind="cubic",
    fill_value=None,
    bounds_error=False,
)
data_contour_N = f(indexy__new_N, amplix__new_N)

# %%
(
    ampli_min_asimov_N,
    ampli_max_asimov_N,
    index_min_asimov_N,
    index_max_asimov_N,
    ampli_best_asimov_N,
    index_best_asimov_N,
) = compute_errors(
    data_contour_N, indexy__new_N, amplix__new_N, threshold_contour, find_min=True
)

threshold_contour = 1

cmap = "plasma"
CS_N = plot_L(
    data_contour_N,
    indexy__new_N,
    amplix__new_N,
    threshold_contour,
    find_min=True,
    color="yellow",
    cmap=cmap,
)
ax = plt.gca()
dat_N = CS_N.allsegs[0][0]
ax.clabel(CS, CS.levels, inline=True, fmt="$1$", fontsize=12)
plt.plot(
    dat[:, 0],
    dat[:, 1],
    color="lightblue",
)
plot_errors(
    ax,
    indexy__new,
    amplix__new,
    ampli_min_asimov,
    ampli_max_asimov,
    index_min_asimov,
    index_max_asimov,
    ampli_best_asimov,
    index_best_asimov,
    "lightblue",
)
plt.plot(index_best_asimov_N, ampli_best_asimov_N, "x", color="yellow")
plt.plot(index_best_asimov, ampli_best_asimov, "x", color="lightblue")
fig = plt.gcf()
if savefig:
    fig.savefig(f"plots/7c_L_contour_{shift}." + picformat)


# %% [markdown]
# ## Compute fraction of within


# %%
# std best fit, nui best fit, ...
within_norm = [
    fraction_within(
        valuies_asimov[0][0] - valuies_asimov[0][1],
        valuies_asimov[0][0] + valuies_asimov[0][1],
        norms,
    ),
    fraction_within(
        valuies_asimov_N[0][0] - valuies_asimov_N[0][1],
        valuies_asimov_N[0][0] + valuies_asimov_N[0][1],
        norms,
    ),
    0.0,
    0.0,
]

within_amplitude = [
    fraction_within(
        valuies_asimov[1][0] - valuies_asimov[1][1],
        valuies_asimov[1][0] + valuies_asimov[1][1],
        amplitudes,
    ),
    fraction_within(
        valuies_asimov_N[1][0] - valuies_asimov_N[1][1],
        valuies_asimov_N[1][0] + valuies_asimov_N[1][1],
        amplitudes,
    ),
    fraction_within(ampli_min_asimov, ampli_max_asimov, amplitudes),
    fraction_within(ampli_min_asimov_N, ampli_max_asimov_N, amplitudes),
]

within_index = [
    fraction_within(
        valuies_asimov[2][0] - valuies_asimov[2][1],
        valuies_asimov[2][0] + valuies_asimov[2][1],
        indices,
    ),
    fraction_within(
        valuies_asimov_N[2][0] - valuies_asimov_N[2][1],
        valuies_asimov_N[2][0] + valuies_asimov_N[2][1],
        indices,
    ),
    fraction_within(index_min_asimov, index_max_asimov, indices),
    fraction_within(index_min_asimov_N, index_max_asimov_N, indices),
]

withins = [within_norm, within_amplitude, within_index]
print(withins)

# %%
mask = dataset_asimov.mask.data.sum(axis=2).sum(axis=1) > 0
ebins = dataset_asimov.counts.geom.axes[0].center[mask]
asimov, asimov_errors = dataset_asimov.models[0].spectral_model.evaluate_error(ebins)
asimov_N, asimov_errors_N = dataset_asimov_N.models[0].spectral_model.evaluate_error(
    ebins
)

withine = []
withine_N = []

for ie, e in enumerate(ebins):
    xs = []
    xs_N = []
    for f in flux:
        value = f[ie]
        lowerlim = np.all(value <= asimov.value[ie] + asimov_errors.value[ie])
        upperlim = np.all(value >= asimov.value[ie] - asimov_errors.value[ie])
        x = lowerlim & upperlim
        xs.append(x)

        lowerlim = np.all(value <= asimov_N.value[ie] + asimov_errors_N.value[ie])
        upperlim = np.all(value >= asimov_N.value[ie] - asimov_errors_N.value[ie])
        x = lowerlim & upperlim
        xs_N.append(x)

    withine.append(np.count_nonzero(xs) / N)
    withine_N.append(np.count_nonzero(xs_N) / N)
print(withine)
print(withine_N)


# %% [markdown]
# ## RND Likelihood Contours


# %%
def read(name):
    with open(name) as data_file:
        data_loaded = json.load(data_file)
    return data_loaded


# %%
cs = []
cNs = []
for shift_rnd, tilt_rnd in zip(shift_rnds, index_rnds):
    try:
        c = read(f"data/contours/6P_068_{shift_rnd:.6}_{tilt_rnd:.6}.json")
        c_N = read(f"data/contours/6P_068_N_{shift_rnd:.6}_{tilt_rnd:.6}.json")
        cs.append(c)
        cNs.append(c_N)
    except:
        print("not found for:", shift_rnd, tilt_rnd)


# %%
from scipy.interpolate import interp1d


def interpolate(cc):
    points = np.column_stack((cc["x"], cc["y"]))  # a (nbre_points x nbre_dim) array

    # Linear length along the line:
    distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]

    # Interpolation for different methods:
    interpolations_methods = ["quadratic"]
    alpha = np.linspace(0, 1, 75)

    interpolated_points = {}
    for method in interpolations_methods:
        interpolator = interp1d(distance, points, kind=method, axis=0)
        interpolated_points[method] = interpolator(alpha)

    for method_name, curve in interpolated_points.items():
        c_int = {"x": curve.T[0], "y": curve.T[1]}
    return c_int


import matplotlib.path as mpltPath


def containing_input(cc):
    path = mpltPath.Path(np.column_stack((cc["x"], cc["y"])))
    mask = path.contains_points(points=([(2.3, 1.0)]))
    return mask[0]


# %%
i = -1
containing = 0
containing_N = 0
for c, c_N in zip(cs[:i], cNs[:i]):
    # plt.plot(c_N['x'], np.array(c_N['y']), label='', color = color+'red', marker = 'o', alpha = alpha)

    c = interpolate(c)
    c_N = interpolate(c_N)

    contains = containing_input(c)
    contains_N = containing_input(c_N)

    if contains:
        containing += 1
        color = "dark"
        fmt = "-"
        alpha = 0.7
    else:
        color = ""
        fmt = "--"
        alpha = 0.4
    plt.plot(
        c["x"],
        np.array(c["y"]),
        label="",
        color=color + "green",
        linestyle=fmt,
        alpha=alpha,
    )
    if contains_N:
        color = "dark"
        fmt = "-"
        alpha = 0.7
        containing_N += 1

    else:
        color = ""
        fmt = "--"
        alpha = 0.4
    plt.plot(
        c_N["x"],
        np.array(c_N["y"]),
        label="",
        color=color + "red",
        linestyle=fmt,
        alpha=alpha,
    )

plt.plot(2.3, 1, "^", color="black")
plt.legend(fontsize=9, loc="best")

plt.xlabel("Index")
plt.ylabel("Amplitude [1e-11 TeV-1 s-1 cm-2]")
print(containing / len(cs))
print(containing_N / len(cNs))

# %% [markdown]
# ## 2D Histograms Poission

# %%
N_bins = N_new + 1
N_bins_y = N_new_y + 1

amplitude_bin = np.linspace(amplix__new_N[0], amplix__new_N[-1], N_bins)
index_bin = np.linspace(indexy__new_N[0], indexy__new_N[-1], N_bins_y)

amplitudes_data = valuies[1][:, 0]
indices_data = valuies[2][:, 0]
data_N_new, (abins, ibins) = np.histogramdd(
    (amplitudes_data, indices_data), (amplitude_bin, index_bin)
)
d_N = (data_contour_N < (1 + data_contour_N.min())).reshape(np.shape(data_contour_N))
d = (data_contour < (1 + data_contour.min())).reshape(np.shape(data_contour))
fraction = np.sum(data_N_new * d) / N * 100
fraction_N = np.sum(data_N_new * d_N) / N * 100

data_N_new, (abins, ibins) = np.histogramdd(
    (amplitudes_data, indices_data), (amplitude_bin, index_bin)
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

im = ax1.pcolormesh(ibins, abins, data_N_new * d)
im = ax2.pcolormesh(ibins, abins, data_N_new * d_N)

for ax in (ax1, ax2):
    ax.set_ylabel("Amplitudes")
    ax.set_xlabel("Indices")
    ax.plot(dat[:, 0], dat[:, 1], color="red")
    ax.plot(dat_N[:, 0], dat_N[:, 1], color="green")

ax1.set_title(f"Within $1\\sigma$ : {fraction:.3} \\%")
ax2.set_title(f"Within $1\\sigma$ Nuisance: {fraction_N:.3} \\%")

# %% [markdown]
# ## Contours

# %%
np.shape(data_N_new)

# %%
import json


def read(name):
    with open(name) as data_file:
        data_loaded = json.load(data_file)
    return data_loaded


# %%

c = read("data/5_scan_68.json")
c39 = read("data/5_scan_39.json")

c_N = read("data/5_scan_N_68.json")


# %%
im = ax.pcolormesh(ibins, abins, data_N_new)


# %%
path = mpltPath.Path(np.column_stack((c["x"], c["y"])))
mask = path.contains_points(points=([(2.3, 1.0)]))
mask


# %%
path = mpltPath.Path(np.column_stack((c_N["x"], c_N["y"])))
mask = path.contains_points(points=([(2.3, 1.0)]))
mask


# %%
np.shape(data_N_new)

# %%
plt.plot(
    c["x"], np.array(c["y"]) * 1e-12, ".-", label="Standard 1$\sigma$", color="red"
)
plt.plot(
    c39["x"],
    np.array(c39["y"]) * 1e-12,
    ".-",
    label="Standard 1$\sigma$",
    color="darkred",
)
plt.plot(
    c_N["x"], np.array(c_N["y"]) * 1e-12, ".-", label="NUI 1$\sigma$", color="green"
)


plt.plot(
    dat[:, 0],
    dat[:, 1],
    color="orange",
)

plt.plot(
    dat_N[:, 0],
    dat_N[:, 1],
    color="lightgreen",
)


plt.scatter(indices[:, 0], amplitudes[:, 0], marker=".")

# %% [markdown]
# # End Plots

# %% [markdown]
# ## Random Choice

# %%

s = sns.color_palette("Paired")
s

# %%
blue = "mediumblue", s[0]  # dark, light
green = s[3], s[2]
red = s[5], s[4]
orange = s[7], s[6]
purple = s[9], s[8]

print(s[0])

# %%
config = Dataset_load.load_config()
colors = config["colors"]["one"]

# asimov, asimov+nui, error asimov, error asimov+nui

a1 = blue
a2 = red
colors = [a1[0], a2[0], a1[1], a2[1]]

# rnd, rnd_example, error rnd_example, rnd+nui, rnd_example+nui, error rnd_example+nui
r1 = orange
r2 = purple
colors_rnd = [r1[0], r1[0], r1[1], r2[0], r2[0], r2[0]]

(
    color_rnd,
    color_rnd_example,
    color_rnd_err,
    color_rnd_nui,
    color_rnd_nui_example,
    color_rnd_nui_err,
) = colors_rnd
cc = color_rnd_example

alpha_nui = 0.99
alpha_st = 0.99
legendsscatter = ["Asimov w/o nui.", "Asimov with nui."]
legends = ["", ""]

alpha_rnd = 0.2
alpha_rnd_nui = 0.2

alpha_example = 0.99
alpha_nui_example = 0.6


nbins = 20

# %%
fpd_Ns, fpd_s = [], []
cs, csN = [], []

for s, t in zip(shift_rnds, index_rnds):
    try:
        fluxpoints = FluxPoints.read(
            f"data/fluxpoints/6P_fluxpoints_{s:.6}_{t:.6}.fits"
        )
        fluxpoints_N = FluxPoints.read(
            f"data/fluxpoints/6P_fluxpoints_N_{s:.6}_{t:.6}.fits"
        )
        model_name = f"data/fluxpoints/6P_model_N_{s:.6}_{t:.6}.yaml"
        fluxpoints_Nds = FluxPointsDataset(
            data=fluxpoints_N, models=Models.read(model_name)
        )
        model_name = f"data/fluxpoints/6P_model_{s:.6}_{t:.6}.yaml"
        fluxpoints_ds = FluxPointsDataset(
            data=fluxpoints, models=Models.read(model_name)
        )
        # fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        # plot_choice_sed(fig, ax)
        # c  = read( f'data/contours/6P_068_{s:.6}_{t:.6}.json')
        c = read(f"data/contours/6P_039_{s:.6}_{t:.6}.json")
        # c_N  = read( f'data/contours/6P_068_N_{s:.6}_{t:.6}.json')
        c_N = read(f"data/contours/6P_039_N_{s:.6}_{t:.6}.json")
        cs.append(c)
        csN.append(c_N)
        fpd_Ns.append(fluxpoints_Nds)
        fpd_s.append(fluxpoints_ds)

    except:
        pass


# %%
len(fpd_Ns)


# %%
i = 0
tilt_rnd, shift_rnd = index_rnds[i], shift_rnds[i]
fluxpoints_Nds, fluxpoints_ds = fpd_Ns[i], fpd_s[i]
c, c_N = cs[i], csN[i]


print(shift_rnd, tilt_rnd)

# %%
model_asimov = Models.read("data/5_model.yml")
model_asimov_N = Models.read("data/5_model_nui.yml")


# %%

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
plot_choice_sed(fig, ax, fluxpoints_Nds)


fig.savefig("plots/7c_fluxpoint_example.pdf", bbox_inches="tight")
fig.savefig("plots/7c_fluxpoint_example.png", bbox_inches="tight")

# %%
fig, (axs, ax1) = plt.subplots(1, 2, figsize=(5, 2.4))
plot_rnd_fistribution(fig, (axs, ax1), cc)
if savefig:
    fig.savefig(f"plots/7c_rnd_sys_{shift}." + picformat)
    fig.savefig(f"plots/7c_rnd_sys_{shift}.pdf")

# %%
fig, axs_L = plt.subplots(1, 1, figsize=(5, 5))
plot_randoms(axs_L, fig)
plot_best_fit_errobar(
    axs_L,
    model=fluxpoints_ds.models,
    model_N=fluxpoints_Nds.models,
    colors=[color_rnd_nui_example, color_rnd_example],
    alpha=[0.9, 0.9],
    label=["Example with nui.", "Example without nui."],
)

plt.plot(
    c["x"],
    np.array(c["y"]) * 1e-12,
    label="",
    color=color_rnd_example,
    linestyle="--",
    alpha=alpha,
)
plt.plot(
    c_N["x"],
    np.array(c_N["y"]) * 1e-12,
    label="",
    color=color_rnd_nui_example,
    linestyle="--",
    alpha=alpha,
)

plot_best_fit_errobar(
    axs_L,
    model=model_asimov,
    model_N=model_asimov_N,
    colors=[colors[1], colors[0]],
    alpha=[0.5, 0.5],
    label=["Asimov with nui.", "Asimov without nui."],
)

axs_L.set_ylim(0.65e-12, 1.4e-12)
axs_L.set_xlim(2.2, 2.4)
axs_L.plot(2.3, 1e-12, "^", color="black", markersize=8, label="Input")
axs_L.legend(loc="upper center", ncol=2)

axs_L.set_xlabel("Index ")
axs_L.set_ylabel("Amplitude [1/cm2 s TeV]")


fig.savefig("plots/7c_scatter.pdf", bbox_inches="tight")
fig.savefig("plots/7c_scatter.png", bbox_inches="tight")


# %%
def compute_precision(N):
    Z = 1.645
    return Z / np.sqrt(N)


energy_power = 2
fig, axs = plt.subplots(1, 1, figsize=(4, 3))
plot_asimov_spectrum(fig, axs)
fp_asimov_N.data.plot(
    energy_power=2,
    color=colors[1],
    ax=axs,
    capsize=(4),
)
fp_asimov.data.plot(
    energy_power=2,
    color=colors[0],
    ax=axs,
    capsize=(4),
)

axs.set_xlim(0.3, 100)
fig.savefig("plots/7c_asimov.pdf", bbox_inches="tight")
fig.savefig("plots/7c_asimov.png", bbox_inches="tight")


# %%
y2 = 2
fig, axs = plt.subplots(
    2,
    2,
    gridspec_kw={"height_ratios": [3, 5], "width_ratios": [5, 3]},
    figsize=(5.0, 5.0),
)
axs_L = axs[1][0]
axs_i = axs[0][0]
axs_a = axs[1][1]
axs[0, -1].axis("off")

plot_randoms(axs_L, fig)
axs_L.set_ylim(0.7e-12, 1.3e-12)
axs_L.set_xlim(2.2, 2.4)

xlims = [None, amplix__new_N, indexy__new_N]
xlabels = ["Best Fit BKG Norm", "Best Fit Amplitude", "Best Fit Index"]


plot_amplitude(axs_a)
plot_index(axs_i)


plt.tight_layout()
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.08, hspace=0.08)
axs_L.legend(bbox_to_anchor=(1.2, 1.7))


# plt.tight_layout()
if savefig:
    fig.savefig(f"plots/7c_L_contour_{shift}_dist_nui." + picformat)
    fig.savefig(f"plots/7c_L_contour_{shift}_dist_nui." + "png")
print(N)

# %%
energy_power = 2
fig, (axs, ax2) = plt.subplots(
    2, 1, gridspec_kw={"height_ratios": [3, 1]}, sharex=True, figsize=((4, 4))
)
# fp_asimov_N.data.plot(energy_power=2, color=colors[1], ax=axs, capsize = (4), )
# fp_asimov.data.plot(energy_power=2, color=colors[0], ax=axs, capsize = (4), )


plot_fp(
    fp_asimov_N.data,
    ax=axs,
    color=colors[1],
    label="",
    marker="x",
    capsize=5,
)
plot_fp(
    fp_asimov_N.data,
    xerr=True,
    ax=axs,
    color=colors[1],
    label="SED with nui.",
    marker="x",
    elinewidth=0.6,
)
plot_fp(
    fp_asimov.data,
    energy_power=2,
    ax=axs,
    color=colors[0],
    label="",
    marker="x",
    capsize=5,
)
plot_fp(
    fp_asimov.data,
    energy_power=2,
    ax=axs,
    xerr=True,
    color=colors[0],
    label="SED w/0 nui.",
    marker="x",
    elinewidth=0.6,
)


plot_asimov_spectrum(fig, axs)
plot_coverage(fig, ax2)
fig.savefig("plots/7c_coverage.pdf", bbox_inches="tight")
fig.savefig("plots/7c_coverage.png", bbox_inches="tight")

# %% [markdown]
# ## Input Within Rnd Results

# %%
a_within, i_within = [], []
a_within_N, i_within_N = [], []

a_input = dataset_asimov.models.parameters["amplitude"].value
i_input = dataset_asimov.models.parameters["index"].value

for a_ in amplitudes:
    a_within.append(((a_[0] - a_[1]) <= a_input) and ((a_[0] + a_[1]) >= a_input))
for i_ in indices:
    i_within.append(((i_[0] - i_[1]) <= i_input) and ((i_[0] + i_[1]) >= i_input))

for a_ in amplitudes_N:
    a_within_N.append(((a_[0] - a_[1]) <= a_input) and ((a_[0] + a_[1]) >= a_input))
for i_ in indices_N:
    i_within_N.append(((i_[0] - i_[1]) <= i_input) and ((i_[0] + i_[1]) >= i_input))


# %%
print(sum(a_within) / len(a_within))
print(sum(i_within) / len(i_within))


# %%
print(sum(a_within_N) / len(a_within_N))
print(sum(i_within_N) / len(i_within_N))


# %%
ampl_range = [0.7, 1.3]
index_range = [1.9, 2.7]


# %%
alpha = 0.7
nnbins = 30
fig, ax = plt.subplots(2, 2, figsize=(9, 3 * 5))
ax = ax.flatten()
nbins = np.linspace(ampl_range[0], ampl_range[1], nnbins)
i = 0
for a_ in amplitudes[:100]:
    within = ((a_[0] - a_[1]) <= a_input) and ((a_[0] + a_[1]) >= a_input)
    if within:
        col = color_rnd
    else:
        col = "red"
    ax[0].errorbar(x=a_[0], y=i, xerr=a_[1], color=col, fmt="x")
    i += 1

i = 0
for a_ in amplitudes_N[:100]:
    within = ((a_[0] - a_[1]) <= a_input) and ((a_[0] + a_[1]) >= a_input)
    if within:
        col = color_rnd_nui
    else:
        col = "red"
    ax[1].errorbar(x=a_[0], y=i, xerr=a_[1], color=col, fmt="x")
    i += 1


ax[0].vlines(1e-12, 0, ax[0].get_ylim()[1], color="black", alpha=alpha)
ax[1].vlines(1e-12, 0, ax[0].get_ylim()[1], color="black", alpha=alpha)
ax[0].set_xlim(ampl_range[0] * 1e-12, ampl_range[1] * 1e-12)
ax[1].set_xlim(ampl_range[0] * 1e-12, ampl_range[1] * 1e-12)
ax[0].set_xlabel("Amplitude [1/TeVcm2s]")
ax[1].set_xlabel("Amplitude [1/TeVcm2s]")

i = 0
for a_ in indices[:100]:
    within = ((a_[0] - a_[1]) <= i_input) and ((a_[0] + a_[1]) >= i_input)
    if within:
        col = color_rnd
    else:
        col = "red"
    ax[2].errorbar(x=a_[0], y=i, xerr=a_[1], color=col, fmt="x")
    i += 1

i = 0
for a_ in indices_N[:100]:
    within = ((a_[0] - a_[1]) <= i_input) and ((a_[0] + a_[1]) >= i_input)
    if within:
        col = color_rnd_nui
    else:
        col = "red"
    ax[3].errorbar(x=a_[0], y=i, xerr=a_[1], color=col, fmt="x")
    i += 1


ax[2].vlines(2.3, 0, ax[0].get_ylim()[1], color="black", alpha=alpha)
ax[3].vlines(2.3, 0, ax[0].get_ylim()[1], color="black", alpha=alpha)
ax[2].set_xlim(index_range[0], index_range[1])
ax[3].set_xlim(index_range[0], index_range[1])
ax[2].set_xlabel("Index []")
ax[3].set_xlabel("Index []")


for a in ax:
    a.legend(title="Containing Input\n Within Errors:", loc="lower right")
plt.tight_layout()

# %%

# %%

# %%
