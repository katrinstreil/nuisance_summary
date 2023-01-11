# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from regions import CircleSkyRegion
import matplotlib.pyplot as plt
from gammapy.data import GTI
from gammapy.irf import EDispKernelMap, EDispMap, PSFKernel, PSFMap
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel
from gammapy.stats import (
    CashCountsStatistic,
    WStatCountsStatistic,
    cash,
    cash_sum_cython,
    get_wstat_mu_bkg,
    wstat,
)
from gammapy.utils.fits import HDULocation, LazyFitsData
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.table import hstack_columns
from gammapy.datasets.core import Dataset
from gammapy.datasets.evaluator import MapEvaluator
from gammapy.datasets.utils import get_axes
from gammapy.datasets import MapDataset

class MapDatasetNuisanceE(MapDataset):
    stat_type = "cash"
    tag = "MapDataset"
    counts = LazyFitsData(cache=True)
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)
    background = LazyFitsData(cache=True)
    psf = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    mask_safe = LazyFitsData(cache=True)

    _lazy_data_members = [
        "counts",
        "exposure",
        "edisp",
        "psf",
        "mask_fit",
        "mask_safe",
        "background",
    ]

    def __init__(
        self,
        models=None,
        counts=None,
        exposure=None,
        background=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        gti=None,
        meta_table=None,
        name=None,
        N_parameters=None,
        penalty_sigma=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}

        self.counts = counts
        #new
        self.N_parameters = N_parameters
        self.penalty_sigma = penalty_sigma
        self.exposure = exposure
        
            
        self.background = background
        self._background_cached = None
        self._background_parameters_cached = None

        self.mask_fit = mask_fit

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )

        self.psf = psf

        if edisp and not isinstance(edisp, (EDispMap, EDispKernelMap, HDULocation)):
            raise ValueError(
                f"'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' object, got {type(edisp)}"
            )

        self.edisp = edisp
        self.mask_safe = mask_safe
        self.gti = gti
        self.models = models
        self.meta_table = meta_table
    
    @property
    def exposure_modified(self):
        if self.N_parameters is not None:
            #print("exposure after ", (self.exposure * (1+self.N_parameters[0].value)).data.max(),
            #     "N: ", self.N_parameters[0].value)

            return  self.exposure * (1+self.N_parameters[0].value)
        else:
            return  self.exposure 
        
    def npred_signal(self, model_name=None):
        """Model predicted signal counts.

        If a model name is passed, predicted counts from that component are returned.
        Else, the total signal counts are returned.

        Parameters
        ----------
        model_name: str
            Name of  SkyModel for which to compute the npred for.
            If none, the sum of all components (minus the background model)
            is returned

        Returns
        -------
        npred_sig: `gammapy.maps.Map`
            Map of the predicted signal counts
        """
        npred_total = Map.from_geom(self._geom, dtype=float)
        #self.update_exposure()
        evaluators = self.evaluators
        if model_name is not None:
            evaluators = {model_name: self.evaluators[model_name]}

        for evaluator in evaluators.values():
            if evaluator.needs_update or self.N_parameters is not None:
                evaluator.update(
                    self.exposure_modified,
                    self.psf,
                    self.edisp,
                    self._geom,
                    self.mask_image,
                )

            if evaluator.contributes:
                npred = evaluator.compute_npred()
                npred_total.stack(npred)

        return npred_total
  
        
    
    def npred(self):
        """Total predicted source and background counts

        Returns
        -------
        npred : `Map`
            Total predicted counts
        """
        npred_total = self.npred_signal()
        if self.background:
            npred_total += self.npred_background() * (1+self.N_parameters[0].value)
        npred_total.data[npred_total.data < 0.0] = 0
        
        return npred_total
    
    
    def stat_sum(self):
        """Total likelihood given the current model parameters."""
        counts, npred = self.counts.data.astype(float), self.npred().data
        if self.penalty_sigma and self.penalty_sigma>0:
            penalty = self.N_parameters[0].value**2 / self.penalty_sigma **2
        else:
            penalty = 0
        #print('penalty', penalty)
        if self.mask is not None:
            return cash_sum_cython(counts[self.mask.data], npred[self.mask.data])  +penalty
        else:
            return cash_sum_cython(counts.ravel(), npred.ravel()) +penalty
        
        
    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.
        name : str
            Name of the sliced dataset.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Sliced dataset

        Examples
        --------
        >>> from gammapy.datasets import MapDataset
        >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        >>> slices = {"energy": slice(0, 3)} #to get the first 3 energy slices
        >>> sliced = dataset.slice_by_idx(slices)
        >>> print(sliced.geoms["geom"])
        WcsGeom
                axes       : ['lon', 'lat', 'energy']
                shape      : (320, 240, 3)
                ndim       : 3
                frame      : galactic
                projection : CAR
                center     : 0.0 deg, 0.0 deg
                width      : 8.0 deg x 6.0 deg
                wcs ref    : 0.0 deg, 0.0 deg
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.slice_by_idx(slices=slices)

        if self.background is not None and self.stat_type == "cash":
            kwargs["background"] = self.background.slice_by_idx(slices=slices)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.psf is not None:
            kwargs["psf"] = self.psf.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)
            
        if self.N_parameters is not None:
            kwargs["N_parameters"] = self.N_parameters

        return self.__class__(**kwargs)
