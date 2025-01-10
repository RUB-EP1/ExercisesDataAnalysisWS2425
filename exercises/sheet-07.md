# Sheet 7: Adaptive binning and weighting

After selecting the data for analysis, it is a common task to weight simulated signal samples to match observed distributions in data.
This is done, since the simulation samples are used further in the measurement (e.g. for efficiency correction), and because they do not describe the data perfectly.

In this exercise, you will explore methods to perform the weighting task in a statistically optimized way.

## Exercise 1

Use the two-dimensional distribution of `log(Lc_PT)` and `K_PID_K` of the preprocessed signal sample for training the classifier (`sheet-06_signal_sample_training.root`, see tutorial 6), to find an optimal binning.
For that, train a `DecisionTreeRegressor` from `DecisionTree.jl` to find a binning with 128 (approximately) equally populated bins.

In view of the next tasks, you will need to define the minimum and maximum of both `log(Lc_PT)` and `K_PID_K`, and implement those as selection requirement.

Make a plot of the resulting binning scheme, drawing lines at all binedges.

## Exercise 2

Using the adaptive binning of Exercise 1, weight the 2D `log(Lc_PT):K_PID_K` distribution of the signal training sample to match the distribution in the data to which you applied your classifier in the previous sheet (`sheet-06_sample-<i>.root`).

Make a plot that compares the distributions of data and signal training sample in one dimensions before the weighting to the weighted signal training distribution.

Make a plot of the distribution of weights.
