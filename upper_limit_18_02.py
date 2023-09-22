import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


class upper_limit:
    def __init__(self, likelihood, mass, scale, name="scale_scan"):
        self.likelihood = likelihood
        self.mass = float(mass)
        self.scale = float(scale)
        self.name = name

    def plot(self):
        fig = plt.figure()
        values = self.likelihood[self.name]
        likelihood_values = self.likelihood["stat_scan"]
        plt.plot(values, likelihood_values, "x", label="mass = " + str(int(self.mass)))
        plt.xlabel(self.name)
        plt.ylabel("stat_scan")
        plt.legend()
        return fig

    def plot_interpolate(self):
        f, xnew = self.interpolate()
        fig = plt.figure()
        values = self.likelihood[self.name]
        likelihood_values = self.likelihood["stat_scan"]
        plt.plot(values, likelihood_values, "x", label="mass = " + str(int(self.mass)))
        plt.plot(xnew, f(xnew), label="interpolated")
        plt.xlabel(self.name)
        plt.ylabel("stat_scan")
        plt.legend()
        return fig

    def interpolate(self):
        xnew = np.linspace(
            self.likelihood[self.name][0], self.likelihood["scale_scan"][-1], 3000
        )
        f = interp1d(
            self.likelihood[self.name], self.likelihood["stat_scan"], kind="cubic"
        )
        return f, xnew

    def parabola(self, x, x_0, sigma):
        return (x - x_0) ** 2 / sigma**2

    def likelihood_upper_extra(self, from_0=True):
        # same as likelihood_upper but extrapolate to outer values in case it was poorly choosen

        # fitting the original data:
        import scipy

        lik = (self.likelihood[self.name], self.likelihood["stat_scan"])
        min_i = np.where(lik[1] == lik[1].min())[0][0]
        sigma_estimate = np.sqrt((lik[0][0] - lik[0][min_i]) ** 2 / lik[1][0])
        params = [
            np.round(lik[0][min_i], int(np.abs(np.log10(lik[0][min_i])) + 2)),
            sigma_estimate,
        ]

        xnew = np.linspace(
            self.likelihood[self.name][0] * 0.99,
            self.likelihood[self.name][-1] * 1.01,
            3000,
        )
        # plt.plot(xnew,  self.parabola(xnew, params[0], params[1]), label = "initial guess")

        ynew = self.parabola(xnew, params[0], params[1])
        fit_params, pcov = scipy.optimize.curve_fit(
            self.parabola, lik[0], lik[1], params
        )
        # str_ = "\n$\sigma $ = {:.2}".format(fit_params[1])
        ynew = self.parabola(xnew, fit_params[0], fit_params[1])
        i_min = np.where(ynew == ynew.min())
        i_min = i_min[0]
        # plt.plot(l[0], l[1], 'x', label = str_  , )
        # plt.plot(xnew,ynew , label = "best fit")
        # plt.legend()

        # try:
        if from_0 == True:
            while xnew[i_min] < 0:
                i_min = i_min + 1
        i_u = i_min[0]
        while ynew[i_u] < (ynew[i_min] + (2.71)):  # 2.71/2.
            i_u = i_u + 1
        i_error = i_min[0]
        while ynew[i_error] < (ynew[i_min] + 1.0):
            i_error = i_error + 1
        # print("minimum at ",xnew[i_min][0], " pm " , xnew[i_error]- xnew[i_min][0])
        # print("upperlimit = ", xnew[i_u])
        # print("returns tupel: minimum, 1sigma error, UL")
        return xnew[i_min], xnew[i_error] - xnew[i_min], xnew[i_u]

        # except:
        #    print("An exception occurred")
        #    return float("nan"), float("nan"), float("nan")

    def likelihood_upper(self, from_0=True):
        f, xnew = self.interpolate()
        i_min = np.where(f(xnew) == f(xnew).min())
        i_min = i_min[0]
        # try:
        if from_0 == True:
            while xnew[i_min] < 0:
                i_min = i_min + 1
        i_u = i_min[0]

        while f(xnew)[i_u] < (f(xnew)[i_min] + (2.71)):  # 2.71/2.
            i_u = i_u + 1
        i_error = i_min[0]
        while f(xnew)[i_error] < (f(xnew)[i_min] + 1.0):
            i_error = i_error + 1
        return xnew[i_min], xnew[i_error] - xnew[i_min], xnew[i_u]

        # except:
        #    print("An exception occurred")
        #    return float("nan"), float("nan"), float("nan")

    def likelihood_error(self, from_0=True):
        f, xnew = self.interpolate()
        i_min = np.where(f(xnew) == f(xnew).min())
        i_min = i_min[0]
        # try:
        if from_0 == True:
            while xnew[i_min] < 0:
                i_min = i_min + 1
        # i_u = i_min[0]
        i_error = i_min[0]
        while f(xnew)[i_error] < (f(xnew)[i_min] + 1.0):
            i_error = i_error + 1
        return xnew[i_min], xnew[i_error] - xnew[i_min]
        # except:
        #    print("An exception occurred")
        #    return float('nan'), float('nan'),float('nan')

    def likelihood_error_asymmetric(self):
        f, xnew = self.interpolate()
        i_min = np.where(f(xnew) == f(xnew).min())
        i_min = i_min[0]
        i_error_neg = i_min[0]
        i_error_pos = i_min[0]
        while f(xnew)[i_error_pos] < (f(xnew)[i_min] + 1.0):
            i_error_pos = i_error_pos + 1
        while f(xnew)[i_error_neg] < (f(xnew)[i_min] + 1.0):
            i_error_neg = i_error_neg - 1
        if i_error_neg < 0:
            i_error_neg = 0
            print("Caution: Neg Error not found! Set to min value!")
        return (
            xnew[i_min],
            xnew[i_min] - xnew[i_error_neg],
            xnew[i_error_pos] - xnew[i_min],
        )

    def plot_upper_limit(self, from_0=False):
        fig, ax = plt.subplots(1, 1)
        # fig.set_figheight(7)
        # fig.set_figwidth(7)
        values = self.likelihood[self.name]
        likelihood_values = self.likelihood["stat_scan"]
        plt.plot(values, likelihood_values, "o", label="mass = " + str((self.mass)))
        plt.xlabel("scale")
        plt.ylabel(self.name)

        x_min, dx_min, x_upper = self.likelihood_upper(from_0=from_0)
        if from_0 == False:
            distance_ul = np.abs(x_min - x_upper)[0]
            print(distance_ul)
        print(dx_min)
        plt.axvline(
            x=x_min,
            linestyle="dashed",
            label="Minimum at {:.3} pm {:.3}".format(x_min[0], dx_min[0]),
        )
        f, xnew = self.interpolate()
        values = self.likelihood["scale_scan"]
        likelihood_values = self.likelihood["stat_scan"]
        plt.plot(
            xnew,
            f(xnew),
        )

        if from_0:
            plt.axvline(
                x=x_upper,
                color="g",
                linestyle="dashed",
                label="Upper limit {:.3}".format(x_upper),
            )
        else:
            plt.axvline(
                x=x_upper,
                color="g",
                linestyle="dashed",
                label="Upper limit {:.3}".format(x_upper),
            )
            plt.axvline(
                x=distance_ul,
                color="blue",
                linestyle="dashed",
                label="Upper limit Projected{:.3}".format(distance_ul),
            )

        plt.axvspan(
            x_min - dx_min,
            x_min + dx_min,
            alpha=0.5,
            color="lightblue",
            label="1 sigma error",
        )
        plt.xlabel("Parameter of Interest")
        plt.ylabel("-2 Log Likelihood")
        plt.legend()
        return fig, ax

    def plot_upper_limit_two(self, likelihood2, mass2, likelihood_upper2):
        fig = plt.figure()
        # fig.set_figheight(10)
        # fig.set_figwidth(10)
        values = self.likelihood[self.name]
        likelihood_values = np.array(self.likelihood[self.name]) + min(
            self.likelihood["stat_scan"]
        )
        values2 = likelihood2[self.name]
        likelihood_values2 = np.array(likelihood2[self.name]) + min(
            likelihood2["stat_scan"]
        )

        plt.plot(
            values,
            likelihood_values,
            "x",
            label="mass = " + str(int(self.mass)) + " GeV",
            color="b",
        )
        plt.plot(
            values2,
            likelihood_values2,
            "x",
            label="mass = " + str(int(mass2)) + " GeV Sim",
            color="r",
        )

        plt.xlabel("Norm")
        plt.ylabel("- 2 Log Likelihood")

        x_min, dx_min, x_upper = self.likelihood_upper()
        x_min2, dx_min2, x_upper2 = likelihood_upper2
        # plt.xlim(-2,20)
        # plt.ylim(-10,80)

        print(dx_min)
        plt.axvline(
            x=x_upper,
            color="b",
            linestyle="dashed",
            label="Upper limit {:.3} Fit".format(x_upper),
        )
        plt.axvline(
            x=x_upper2,
            color="r",
            linestyle="dashed",
            label="Upper limit {:.3} Sim".format(x_upper2),
        )

        plt.xlabel("scale")
        plt.ylabel("likelihood")
        # plt.titel("mass_DM = 2TeV")
        plt.legend()
        return fig
