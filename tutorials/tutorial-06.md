# Tutorial 6: Classification

## Inspect input data

Before training a classifier, it makes sense to inspect the input data carefully. <br>
In fact, for most tasks that have to do with machine learning, working on the input data is the most important and time-consuming step. <br>

We will work again with the data resembling the $\Xi^- K^+ \pi^+$ final state.<br>
You are provided with signal and background samples that you will use for training a classifier that will discriminate signal from background.<br>

In this tutorial, the task is to inspect these samples, make plots, filter and transform the variables that you are going to use for the training. This step is often also called pre-processing.<br>
Most of the pre-processing work, like choosing data and selecting the variables, has already been done for you, and you can concentrate on the final steps.<br>

First, show normalized signal and background distributions for each variable in the same plot.

> [!TIP]
> Use a loop over the column names of a julia dataframe (calling `names(dataframe)`).<br>
> When you encounter issues with plotting in a loop, you can define a plot outside the loop `plt = plot(layout=(17,1),size=(width=640,height=4000))`, and fill the sub-plots in the loop `plot!(plt, ...)`.

The first set of plots will most likely look a bit rough. If you see spikes in distributions, it might make sense to take the logarithm of the variable.<br>
You can for example put the variable names that you would like to transform in a list and add an `if` statement in the loop.

You may also want to work on a range for every plot, such that signal and background distributions use the same range and binning.

> [!TIP]
> Take a look at the `describe` function to get the plot ranges.

After that, you may see that some variables have sharp cutoffs in the background-distributions, but not in signal.<br>
This is, because a selection was applied to the background sample, that is also present in the data you will later apply the classifier to.<br>
You will need to filter the rows from the dataframe of the signal distributions where this happens. The selections are

```sh
Lc_DOCA_<ij><0.2
pi_OWNPVIPCHI2>4
pi_PT>280
```

Last, but not least, there are outliers in the `pi_PID_K` distribution. They can be removed with e.g. `pi_PID_K>-900`.
