class MapDataset(Dataset):

    def __init__(
        self,
        e_reco_n=2000,
        models=None,
        counts=None,
        edisp=None,
    ):
        self._evaluators = {}
        self.counts = counts
        self.exposure = exposure
        self.background = background
        self._background_cached = None
        self._background_parameters_cached = None
        self._irf_parameters_cached = None
        self._irf_cached = None

        self.mask_fit = mask_fit

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )

        self.edisp = edisp
        self.models = models
        self.e_reco_n = e_reco_n

    # TODO: keep or remove?
    @property
    def background_model(self):
        try:
            return self.models[f"{self.name}-bkg"]
        except (ValueError, TypeError):
            pass

    @property
    def irf_model(self):
        try:
            return self.models[f"{self.name}-irf"]
        except (ValueError, TypeError):
            pass
    @property
    def models(self):
        """Models set on the dataset (`~gammapy.modeling.models.Models`)."""
        return self._models
    
    @models.setter
    def models(self, models):
        """Models setter"""
        self._evaluators = {}

        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)

            for model in models:
                if not isinstance(model, FoVBackgroundModel) and not isinstance(
                    model, IRFModels
                ):
                    evaluator = MapEvaluator(
                        model=model,
                        evaluation_mode=EVALUATION_MODE,
                        gti=self.gti,
                        use_cache=USE_NPRED_CACHE,
                    )
                    self._evaluators[model.name] = evaluator
        self._models = models

    @property
    def evaluators(self):
        """Model evaluators."""
        return self._evaluators
    

    def npred(self):
        """Total predicted source and background counts.

        Returns
        -------
        npred : `Map`
            Total predicted counts
        """
        npred_total = self.npred_signal()

        if self.background:
            npred_total += self.npred_background()
        npred_total.data[npred
                         _total.data < 0.0] = 0
        return npred_total


    def edisp_helper(self, energy):
        energy_rebins = MapAxis(
            nodes=np.logspace(
                np.log10(energy.center[0].value),
                np.log10(energy.center[-1].value),
                self.e_reco_n * len(energy.center),
            ),
            node_type="center",
            name="energy",
            unit="TeV",
            interp="log",
        )
        return energy_rebins

    def npred_edisp(self):
        """Predicted edisp map
        Returns
        -------
        irf : `Map`
        """
        # print("npred_edisp")
        edisp = self.edisp
        # get the kernel
        edisp_kernel = edisp.get_edisp_kernel()
        # rebin enenergyaxis
        energy_rebins = self.edisp_helper(edisp_kernel.axes["energy"])
        # compute gaussian with new eaxis
        gaussian = self.irf_model.e_reco_model(
            energy_axis=energy_rebins,
        )
        # rebin edisp_kernel data and multiply with gaussian
        data_rebinned = np.matmul(
            np.repeat(
                edisp_kernel.data,
                self.e_reco_n,
                axis=1,
            ),
            gaussian,
        )
        # set as kernel data
        edisp_kernel.data = data_rebinned.reshape(
            (
                len(edisp_kernel.axes["energy_true"].center),
                1,
                len(edisp_kernel.axes["energy"].center),
                self.e_reco_n,
            )
        ).mean(axis=(1, 3))

        edisp = EDispKernelMap.from_edisp_kernel(edisp_kernel)
        return edisp

    def npred_background(self):
        """Predicted background counts.

        The predicted background counts depend on the parameters
        of the `FoVBackgroundModel` defined in the dataset.

        Returns
        -------
        npred_background : `Map`
            Predicted counts from the background.
        """
        background = self.background
        if self.background_model and background:
            if self._background_parameters_changed:
                values = self.background_model.evaluate_geom(geom=self.background.geom)
                if self._background_cached is None:
                    self._background_cached = background * values
                else:
                    self._background_cached.quantity = (
                        background.quantity * values.value
                    )
            return self._background_cached
        else:
            return background

        return background

    def _irf_parameters_changed(self):
        values = self.irf_model.parameters.value
        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._irf_parameters_cached == values)
        if changed:
            self._irf_parameters_cached = values
        return changed

    def _background_parameters_changed(self):
        values = self.background_model.parameters.value
        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._background_parameters_cached == values)
        if changed:
            self._background_parameters_cached = values
        return changed

    @deprecated_renamed_argument("model_name", "model_names", "1.1")
    def npred_signal(self, model_names=None, stack=True):
        """Model predicted signal counts.

        If a list of model name is passed, predicted counts from these components are returned.
        If stack is set to True, a map of the sum of all the predicted counts is returned.
        If stack is set to False, a map with an additional axis representing the models is returned.

        Parameters
        ----------
        model_names: list of str
            List of name of  SkyModel for which to compute the npred.
            If none, all the SkyModel predicted counts are computed
        stack: bool
            Whether to stack the npred maps upon each other.

        Returns
        -------
        npred_sig: `gammapy.maps.Map`
            Map of the predicted signal counts
        """
        npred_total = Map.from_geom(self._geom, dtype=float)

        evaluators = self.evaluators
        if model_names is not None:
            if isinstance(model_names, str):
                model_names = [model_names]
            evaluators = {name: self.evaluators[name] for name in model_names}

        npred_list = []
        labels = []
        for evaluator_name, evaluator in evaluators.items():
            if evaluator.needs_update:
                evaluator.update(
                    self.exposure,
                    self.psf,
                    self.edisp,
                    self._geom,
                    self.mask_image,
                )

            if self.irf_model is not None and self._irf_parameters_changed():
                edisp = self.edisp
                exposure = self.exposure

                if self.irf_model.e_reco_model is not None:
                    edisp = self.npred_edisp()
                    evaluator.update_edisp(edisp, self._geom)

                if self.irf_model.eff_area_model is not None:
                    exposure = self.npred_exposure()
                    evaluator.update_exposure(exposure, self.mask_image)

                if self.irf_model.psf_model is not None:
                    evaluator.convolve_psf_map(self.npred_gaussian_psf_map())

                # evaluator.update(
                #    exposure,
                #    self.psf,
                #    edisp,
                #    self._geom,
                #    self.mask_image,
                # )

            if evaluator.contributes:
                npred = evaluator.compute_npred()
                if stack:
                    npred_total.stack(npred)
                else:
                    npred_geom = Map.from_geom(self._geom, dtype=float)
                    npred_geom.stack(npred)
                    labels.append(evaluator_name)
                    npred_list.append(npred_geom)

        if npred_list != []:
            label_axis = LabelMapAxis(labels=labels, name="models")
            npred_total = Map.from_stack(npred_list, axis=label_axis)

        return npred_total
    
    
dataset =MapDataset()
dataset.models  = Models()
npred = dataset.npred_signal()