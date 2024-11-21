# Sheet 4: Hypothesis testing

It is now time to estimate whether the hypothesized signal for the $\Xi_c^+ \rightarrow \Xi^- K^+ \pi^+$ decay process is statistically significant.

## Exercise 1

Given your dataset, estimate the $p$-value of the background-only hypothesis ($p_0$), using a test statistic with the Poisson distribution. The number of signal and background events should be computed from the central 1 $\sigma$ interval of the Gaussian signal.

## Exercise 2

Compute the same $p$-value with a simple likelihood-ratio test statistic $Q=\frac{L_{s+b}(\mu=1)}{L_{b}(\mu=0)}$,
where $\mu$ indicates the signal strength, with $\mu=1$ being that of the best fit result.

The distribution of the test statistic should be obtained by sampling datasets under the background-only hypothesis and evaluating the likelihood on this "toy" dataset with the signal + background hypothesis. Note that all model parameters are fixed for the evaluation of the test statistics.

## Exercise 3

At the time of the Higgs discovery, the ATLAS and CMS collaborations used the "Brazil flag" plot to show the significance of the signal (see e.g. [ATLAS figure from CERN library reference](https://cds.cern.ch/record/1471031/files/CombinedResults.png)). <br>
In this exercise, we break down technical details of the construction.

In the top plot, you see 95 % confidence limits on the standard model signal strength computed with the $\mathrm{CL}_s$ method as a function of the reconstructed invariant-mass. The $\mathrm{CL}_s$ method has been developed for Higgs searches at LEP, the predecessor of the LHC, and was using the simple likelihood-ratio test statistic $Q$ instead of the now more common profile likelihood ratio.

Your task is to compute the expected 95 % $\mathrm{CL}_s$ limits in 20 points of the $\Xi_c^+$ signal fit fraction up to twice the signal strength that you have observed in the fit.

> [!TIP]
> Since $\mathrm{CL}_s(\theta) = \frac{p_1(\theta)}{1-p_0}$ is defined as the ratio of $p$-values of rejecting the alternative hypothesis ($p_1$)
> over accepting the null-hypothesis ($1-p_0$), you need to generate toy datasets (about 500 should be sufficient) for a given alternative hypothesis,
> compute the test statistic and find the central quantiles of the test statistic.

## Exercise 4* (extra points)

Perform a scan of the mass parameter in 10 bins around the observed signal to compute Observed and Background expected CLs limits similar to the "Brazil flag" plot of ATLAS.
