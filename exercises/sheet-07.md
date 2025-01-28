# Sheet 7: Adaptive binning and weighting

After selecting the data for analysis,
it is a common task to weight simulated signal samples to match observed distributions in data.
This is done, since the simulation samples do not describe the data perfectly, while
having the same distributions on marginalized dimentions is important for tasks like efficiency correction.

In this sheet, you will explore methods to reweight the `log(Lc_PT)` and `K_PID_K` distributions of the Monte Carlo (MC) sample (`sheet-06_signal_sample_training.root`; see [tutorial-06](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/tutorials/tutorial-06.md)) to match the data sample obtained earlier (see [sheet-02](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-02.md)).

## Exercise 1

We will start by iteratively reweighting one-dimensional distributions to observe a convergence process.
The algorithm proceeds as follows:

1. Set initialize weights by assigning uniform values to the MC sample so that its total yield matches the data sample size.

2. Reweight `log(Lc_PT)`  
   - Create histograms of the `log(Lc_PT)` distribution for both the data and the (currently) weighted MC sample.  
   - In each bin, compute a scaling coefficient as the ratio of the data bin content to the MC bin content.  
   - Update the MC weights by multiplying the old weights by the bin-specific scaling coefficient.  
   - Verify that the weighted MC distribution matches the data more closely.

3. Reweight `K_PID_K` by repeating the same procedure (histogram, compute scaling coefficients, update weights) for the `K_PID_K` distribution.

4. Iterate steps 2 and 3 multiple times to see incremental improvements.

5. Analyze convergence by tracking and reporting how the mean, standard deviation, and correlation of the weighted MC sample change over these iterations, and compare with the corresponding data values.


## Exercise 2

As discussed at the lecture (see [lecture-14b](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/lectures/lecture-14-b.jl)), a binary classifier can be used for efficient reweighting of complex distributions.
- Train a gradient-boosted decision tree with a logistic loss function on the two-dimensional `log(Lc_PT):K_PID_K` distribution, using label `1` for the data and `0` for the MC sample.
- Compute the weights as `p / (1 - p)`, where `p` is the classifier’s predicted probability for the data label.
- Compare the one-dimensional distributions (before and after weighting) for data and the signal training sample.
- Compute the mean, standard deviation, and correlation of the weighted MC sample for different `max_depth` values (e.g., 4, 5, and 6).
