# Sheet 6: Classification

The samples that you have received for Sheet 2 still contains a large amount of (combinatorial) background events. <br>
In this sheet, you will use Machine Learning to classify signal and background events from your enhanced data-like sample (you should have received an email with a link to a new `root` file). <br>
For the classification task, you will use proxies for signal and background data, that provided centrally through moodle.
These samples contain events that are mostly of combinatorial nature for the background (in this case from a so called high-mass sideband), and simulated signal events for the signal.

## Exercise 1

Using the pre-processed signal and background samples from tutorial 6, train 3 classifiers of your choice (different types, or different tuning of hyperparameters) and compare their ROC curves measured on the training data. Argue which classifier you would choose to apply to your data.

## Exercise 2

Which variables are the most discriminating ones (look for `impurity_importance` in [DecisionTree.jl](https://github.com/JuliaAI/DecisionTree.jl))? Are the highest ranked variables always the same? Print the importance of variables for the classifier you chose in Exercise 1, and make a plot of the 3 highest ranked ones, showing the normalized training data distributions and the distribution of the variable in your dataset.

## Exercise 3

Apply the classifier to your data sample. Scan the classifier response (i.e. select events where the response is larger than the scanned value) to obtain the point where the figure of merit, $S/\sqrt{S+B}$, of the larger peak (the $\Lambda_c^+$ signal) is maximal. As range to compute signal and background integrals, use twice the width of the $\Lambda_c^+$ signal on each side of the signal mean, prior to applying a selection on the classifier response.
Make a plot of the figure of merit as a function of the classifier response.

## Exercise 4 (extra points)

Re-compute the p-value for $\Xi_c^+$ signal.
