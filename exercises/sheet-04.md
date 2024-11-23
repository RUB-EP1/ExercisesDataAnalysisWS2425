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

Compute the same $p$-value with a simple likelihood-ratio test statistic $-2\ln Q$ with $Q=\displaystyle\frac{L_{s+b}(\mu=1)}{L_{b}(\mu=0)}$,
and where $\mu$ indicates the signal strength, with $\mu=1$ being that of the best fit result.

Contrary to the previous exercise, the distribution of the test statistic is not known.
It can be obtained by sampling datasets under the background-only hypothesis and evaluating the likelihood on this "toy" dataset with the signal + background hypothesis.
Note that the definition of the test statistic is such, that all parameters are fixed; i.e. you only need to evaluate the likelihood ratio without having to fit the toy data.

Make a plot of the test statistic distribution and the observed value of the test statistic under the signal+background hypothesis on your initial dataset.

Convert the $p$-value to a significance and compare it to common approximations
$Z\approx\sqrt(-2 \ln Q)$, $Z\approx S/\sqrt(S+B)$, $Z\approx 2(\sqrt{S+B}-\sqrt{B})$ and $Z\approx\sqrt{2(S+B) \ln(1+S/B) - 2S}$, where $S$ and $B$ and the number of signal and background events (in a given interval).

## Exercise 3

At the time of the Higgs discovery, the ATLAS and CMS collaborations used the "Brazil flag" plot to show the significance of the signal
(see e.g. [ATLAS figure from CERN library reference](https://cds.cern.ch/record/1471031/files/CombinedResults.png)). <br>
In this exercise, we break down technical details of the construction.

In the top plot, you see 95 % confidence limits on the standard model signal strength computed with the $\mathrm{CL}_s$ method as a function of the reconstructed invariant-mass.
The $\mathrm{CL}\_{s}$ method has been developed for Higgs searches at LEP [1], the predecessor of the LHC.

Your task is to compute the observed $\mathrm{CL}\_{s}$ values in 20 points of the $\Xi_c^+$ signal fit fraction up to twice the signal strength that you have observed in the fit.

Since $\mathrm{CL}\_{s} ( \theta ) = \displaystyle\frac{ p\_{1} ( \theta ) }{ 1 - p\_{0} }$
is defined as the ratio of $p$-values of rejecting the alternative hypothesis ($p\_{1}$)
over accepting the null-hypothesis ($1-p\_{0}$), you need to generate toy datasets (about 500 should be sufficient) for a given alternative hypothesis.
Evaluate the test statistic $-2\ln Q$ and the integral from the observed value on data.
You can re-use the distribution of the test statistic under the null hypothesis found in Exercise 2.

Make a plot of the observed distribution of $\mathrm{CL}_s$ as a function of $\theta$ to read off the 95 % upper limit on $\theta$.

## Exercise 4* (extra points)

Perform a scan of the mass parameter in 10 bins around the signal peak position to compute observed CLs limits, corresponding to the solid black line of the "Brazil flag" plot of ATLAS.

## References

[1] https://inspirehep.net/literature/599622
