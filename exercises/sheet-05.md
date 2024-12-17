# Sheet 5: Wilks' theorem

Wilks' theorem is widely used in (particle) physics. It is important to know its implications and limitations to properly use it.

## Exercise 1

Generate 3 toy datasets of a flat background model plus a Gaussian signal distribution (the Anka model) with 20, 200 and 2000 entries in total.
Choose the range of support such it is centered around the Gaussian mean, and that the range covered corresponds to 20 times the Gaussian width.
The integrals of signal and background should be equal.

Fit the toy data with the model used for generation for four different scenarios:

1. all parameters except the signal strength[1] are known (fixed to the values used to generate the toy);
2. the signal strength and the width are unknown;
3. signal strength, width and mean of the Gaussian are unknown;
4. all parameters are unknown.

Compute the $p$-value from the likelihood ratio and the $\chi^2$ distribution for each case.

Generate 1000 new toy samples from the best fit results in each case and compare the test statistic distribution to the corresponding $\chi^2$ distribution in a plot.

Discuss the observations.

[1] note that signal strength implies that both the Gaussian `a` and the `flat` parameter of the background are free in the fit (using the definition of the Anka model).
