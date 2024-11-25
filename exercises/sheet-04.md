# Sheet 4: Hypothesis testing

It is now time to estimate whether the hypothesized signal for the $\Xi_c^+ \rightarrow \Xi^- K^+ \pi^+$ decay process is statistically significant.

## Exercise 1

We will start with an unconventional way to estimate a $p$ value (or significance), which should help you to get familiar with the concepts.

Given your dataset, estimate the $p$-value of the background-only hypothesis ($p_0$), using s Poisson distribution as test statistic.
The expectation value for the Poisson distribution is the observed number of background events extracted from a fit without signal.
As this number would depend on the fit range, it should be computed in the central 1 $\sigma$ interval (68.3 percentile) of the Gaussian signal,
when fitting the data under the signal+background hypothesis.<br>
The $p$-value will be integral of the test statistic from the observed number of events in the defined range under the signal+background hypothesis to infinity.

## Exercise 2

Compute the same $p$-value with a simple likelihood-ratio test statistic $-2\ln Q$ with $

$$Q=\displaystyle\frac{L_{s+b}(\mu=1)}{L_{b}(\mu=0)},$$

and where $\mu$ indicates the signal strength, with $\mu=1$ being that of the best fit result.

Contrary to the previous exercise, the distribution of the test statistic is not known.
It can be obtained by sampling datasets under the background-only hypothesis and evaluating the likelihood on this "toy" dataset with the signal + background hypothesis.
Note that the definition of the test statistic is such, that all parameters are fixed; i.e. you only need to evaluate the likelihood ratio without having to fit the toy data.

Make a plot of the test statistic distribution and the observed value of the test statistic under the signal+background hypothesis on your initial dataset.

Convert the $p$-value to a significance and compare it to common approximations
- $Z\approx\sqrt{-2 \ln Q}$,
- $Z\approx S/\sqrt{S+B}$,
- $Z\approx 2(\sqrt{S+B}-\sqrt{B})$, and
- $Z\approx\sqrt{2(S+B) \ln(1+S/B) - 2S}$, where $S$ and $B$ and the number of signal and background events (in a given interval).

## Exercise 3

At the time of the Higgs discovery, the ATLAS and CMS collaborations used the "Brazil flag" plot to show the significance of the signal
(see e.g. [ATLAS figure from CERN library reference](https://cds.cern.ch/record/1471031/files/CombinedResults.png)). <br>
In this exercise, we break down technical details of the construction.

In the top plot, you see 95 % confidence limits on the standard model signal strength computed
with the $\mathrm{CL}_s$ method as a function of the reconstructed invariant-mass.
The $\mathrm{CL}_{s}$ method has been developed for Higgs searches at LEP [1], the predecessor of the LHC.

Your task is to compute one slides of the "Brazil flag" plot for $\Xi_c^+$ distribution performing a scan over the signal strength.

Using default values of the fit obtained in Sheet 3, you can fix all parameters and defined the $H_0$ and $H_1$ hypotheses, as background-only,
and background+signal, respectively.
1. Using an estimation of the signal significance from Exercise 2, find a range of the signal-strength parameter $a$ in which the significance changes from 0.5σ to 2.0σ. Define four scan points in this range.
2. For each value of the strength parameter, generate a distribution of the test statistics defined as $2\ln Q$ using a pseudo datasets under the $H_0$ and $H_1$ hypothesis.
3. Compute $p_0$, $\mathrm{CL}_{s}$, $\mathrm{CL}_{s+b}$, and $\mathrm{CL}_{s} = \mathrm{CL}_{s+b}/\mathrm{CL}_{b}$ for each toy dataset.
4. Plot results of the $\mathrm{CL}_{s}$ scan as a function of the signal strength.
5. Add a value of the $\mathrm{CL}_{s}$ for the observed data.
6. Determine the internal of the strength parameter such that the value $0.05$ is covered within the central 68% of $\mathrm{CL}_{s}$ distribution (use the scan values, and $0.16$ and $0.84$ quantiles of the test statistics distribution under the $H_0$ hypothesis)
7. Find the value of the strength parameter for which the observed $\mathrm{CL}_{s}$ value reaches 0.05.

## Exercise 4* (extra points)

Perform a scan of the mass parameter in 10 bins around the signal peak position to compute observed CLs limits, corresponding to the solid black line of the "Brazil flag" plot of ATLAS.

## References

[1] https://inspirehep.net/literature/599622
