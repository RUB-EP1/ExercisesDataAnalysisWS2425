# Sheet 3: Profile Likelihood and bootstrapping

In this sheet you will learn more about parameter estimation by constructing a profile likelihood ratio and using bootstrapping techniques for finding a confidence interval.

You will continue working with the sample that you have obtained for Sheet 2.

When fitting the $\Xi^- K^+ \pi^+$ invariant-mass distribution (`Lc_M`) for the previous sheet and inspecting the pulls carefully, you may have noticed a second peak at higher mass.<br>
This is exciting! If you can prove that the small enhancement is statistically significant, you have evidence for a decay process that has never been measured: $\Xi_c^+ \rightarrow \Xi^- K^+ \pi^+$.<br>
To work towards that, you will estimate the *yield* (i.e. the integral of the signal p.d.f.) of this process.

## Exercise 1

Add a second Gaussian to your fit model, fit the data again, and compute the Hessian matrix. Use that to compute the uncertainty of the signal yield for the new decay process.
> [!TIP]
> Look at the [pdglive](https://pdglive.lbl.gov/Particle.action?init=0&node=S045&home=BXXX040) website of the $\Xi_c^+$ particle to get a good starting parameter for the mean of your second Gaussian.

## Exercise 2

Compute the profile likelihood ratio in 20 points around the best fit value of the yield. Choose the points to scan based on the uncertainties determined in Exercise 1. How does the two-sided confidence level at 68.3% (1 $\sigma$) constructed with the profile likelihood compare to the uncertainty calculated from the Hessian matrix?

## Exercise 3

Use the jackknife bootstrapping method to estimate the variance of the yield. How does it compare to the previous methods?
