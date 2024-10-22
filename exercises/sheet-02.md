# Sheet 2: Likelihood and fitting

In this sheet you will explore fitting continuous distributions to a data sample and parameter estimation.

You will be studying a sample that resembles the decay process $\Lambda_c^+ \rightarrow \Xi^- K^+ \pi^+$ in the currently running LHCb experiment at CERN.

<details> <summary> Details about the data </summary>

The sample contains three variables: `Lc_M`, `Xi_M` and `L_M`.
They are all so-called [invariant-mass](https://en.wikipedia.org/wiki/Invariant_mass) variables.
While `Lc_M` is the variable you will be fitting, `Xi_M` and `L_M` are the invariant masses of unstable particles in the decay chain, that are used to *reconstruct* the $\Lambda_c^+$ baryon.

Roughly speaking, reconstruction transforms detector responses into properties like e.g. trajectories and momentum of particles that physicists can work with. In this example of a $\Lambda_c^+ \rightarrow \Xi^- K^+ \pi^+$ decay, we reconstruct the $K^+$ and $\pi^+$ particles directly from detector responses, while the $\Xi^-$ particle itself is a weakly decaying particle. It's decay-products are a $\Lambda$ baryon and a $\pi^-$ meson. The $\Lambda$ baryon decays further into a proton and a negatively charged pion. We will be looking at the invariant masses of proton and $\pi^-$ (`L_M`) and the invariant-mass of $\Lambda\pi^-$ (`Xi_M`) to select a cleaner $\Lambda_c^+$ sample.

Due to finite detector resolution, the data coming from the $\Lambda_c^+ \rightarrow \Xi^- K^+ \pi^+$ signal decay is roughly Gauss-distributed. Further, there are so-called combinatorial backgrounds that arise when combining e.g. $\Xi^-$ with $K^+$ and $\pi^+$ coming from different decay processes.

Details about particles and their decays are collected in the ["Review of Particle Physics"](https://pdglive.lbl.gov/Viewer.action). If you click your way through, you can see that the decay process $\Lambda_c^+ \rightarrow \Xi^- K^+ \pi^+$ is listed [here](https://pdglive.lbl.gov/BranchingRatio.action?pdgid=S033.27&home=BXXX040) and has been first observed in 1991.
</details>

## Exercise 1

Use julia dataframes to select events in the $\Lambda$ and $\Xi^-$ signal region (`L_M`$\in[1108,1123]$ and `Xi_M`$\in[1312,1330]$).

Draw a histogram of `Lc_M` with 90 bins in the range 2200 to 2560 (MeV) before and after the selection.

## Exercise 2

Fit the invariant-mass of the $\Xi^- K^+ \pi^+$ system (`Lc_M`) with a polynomial and a Gaussian function. Choose the order of the polynomial, and briefly explain your choice.

## Exercise 3

Draw a graph or histogram that shows the pull of the fit model with respect to the data as a function of `Lc_M`. Briefly describe what you conclude from this graph/histogram. Are there more structures in the spectrum?

Draw a histogram of the pull values. How should this histogram look like for a good fit?

## Exercise 4

Estimate the mass of the $\Lambda_c^+$ baryon and compare it to [the known mass value](https://pdglive.lbl.gov/Particle.action?init=0&node=S033&home=BXXX040).
