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
# Fitting asimov datasets with the nuisance parameter bias

# %% [markdown]
# ### Setup

# %%
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

from regions import CircleSkyRegion, RectangleSkyRegion
import yaml
import sys
sys.path.append('../')
import Dataset_load 

from  Dataset_Setup import Setup, GaussianCovariance_matrix


# %% [markdown]
# ## Methods


# %%
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
    ax.set_ylim(1e-12, 2e-11)


# %% [markdown]
# ## Import

# %%
config = Dataset_load.load_config()



# %% [markdown]
# ### Datasets

# %%
livetimes = np.logspace(-2, 2,  13)[:6] 
livetimes = np.append(livetimes, np.logspace(0, 2, 7))
live = livetimes[7]



# %%
dataset_asimov = Dataset_load.create_asimov(
   model = 'crab', source = "PKSflare",  parameters=None, livetime = f"{live}-hr",
)

# %%
norm = 0.1
tilt = 0.025
bias = 0.0
resolution = 0.0

# %%
setup = Setup(dataset_input=dataset_asimov)
#setup.set_up_irf_sys(bias, resolution, norm, tilt)
dataset_asimov, dataset_asimov_N = setup.run()
# irf model
setup.set_irf_model(dataset_asimov_N)
dataset_asimov_N.models.parameters['resolution'].frozen = True
dataset_asimov_N.models.parameters['bias'].frozen = True

dataset_asimov_N.irf_model.parameters['tilt'].frozen = False
dataset_asimov_N.irf_model.parameters['norm'].frozen = False
setup.set_irf_prior(dataset_asimov_N, bias, resolution, norm, tilt)

# %%
a_unit = dataset_asimov_N.models.parameters["amplitude"].unit

# %%
# %%time
fitting = 1
if fitting:
    fit = Fit(store_trace=False)
    result = fit.run([dataset_asimov])
    result_N = fit.run([dataset_asimov_N])

    dataset_asimov.models.write("data/1_model.yml", overwrite=1)
    dataset_asimov_N.models.write("data/1_model_N.yml", overwrite=1)


else:
    m = Models.read("data/1_model.yml")
    dataset_asimov.models = Models(
        [
            m[0],
            FoVBackgroundModel(
                dataset_name=dataset_asimov.name,
                spectral_model=m[1].spectral_model.copy(),
            ),
        ]
    )
    path = "data/1_model_N.yml"
    dataset_asimov_N = Dataset_load.load_dataset_N(dataset_asimov_N, path)


# %%
dataset_asimov_N

# %%
print("index:")
print(
    dataset_asimov.models.parameters["index"].value,
    "pm",
    dataset_asimov.models.parameters["index"].error,
)
print(
    dataset_asimov_N.models.parameters["index"].value,
    "pm",
    dataset_asimov_N.models.parameters["index"].error,
)


print("lambda:")
print(1 / 60)
print(
    dataset_asimov.models.parameters["lambda_"].value,
    "pm",
    dataset_asimov.models.parameters["lambda_"].error,
)
print(
    dataset_asimov_N.models.parameters["lambda_"].value,
    "pm",
    dataset_asimov_N.models.parameters["lambda_"].error,
)

print("wihtout:",
    1 / dataset_asimov.models.parameters["lambda_"].value,
    "pm",
    dataset_asimov.models.parameters["lambda_"].error
    / dataset_asimov.models.parameters["lambda_"].value ** 2,
)
print("wiht",
    1 / dataset_asimov_N.models.parameters["lambda_"].value,
    "pm",
    dataset_asimov_N.models.parameters["lambda_"].error
    / dataset_asimov_N.models.parameters["lambda_"].value ** 2,
)

# %%
print(dataset_asimov.models)
print(dataset_asimov_N.models)

# %%
# cutoff error in percent:
per = dataset_asimov.models.parameters["lambda_"].error/ dataset_asimov.models.parameters["lambda_"].value
print("without:",per)
per_N = dataset_asimov_N.models.parameters["lambda_"].error/ dataset_asimov_N.models.parameters["lambda_"].value
print("with:   ", per_N)

print("testL", np.sqrt(per**2 + 0.1 ** 2))

# %%

energy_power = 2
fig, axs = plt.subplots(1, 1, figsize=(4, 3))
plot_asimov_spectrum(fig, axs)
axs.set_ylim(1e-12, 1e-10)

axs.set_xlim(0.3, 100)
fig.savefig("plots/asimov_crab.png")

# %%

# %%
