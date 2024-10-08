# Sheet 1: Continuous Distributions

In this sheet we explore continuous distributions and operations with them.<br>
We learn about histogramming and methods of sampling.

## Exercise 1

Write code for the following continuous distributions (from scratch):

a) Gaussian<br>
b) Exponential<br>
c) 3rd order polynomial<br>
d) Relativistic Breit-Wigner (RBW)<br>
e) Voigt profile (convolution of RBW with Gaussian)

Define them such that you can set parameters like mean and width later on.

> [!TIP]
> Read about [functions](https://docs.julialang.org/en/v1/manual/functions/) in the julia manual.

Make a plot for each of the distributions with parameters

a) $\mu=1.0, \sigma=0.4$<br>
b) $\tau=-0.05$<br>
c) $a=0.01, b=-0.04, c=0.25$<br>
d) $M=1530, \Gamma=9$<br>
e) $M=1530, \Gamma=9, \sigma=16$

> [!NOTE]
> You will need to choose a reasonable range for plotting.

> [!TIP]
> Read about [Plots](https://docs.juliaplots.org/latest/) in the julia `Plots` package manual.

## Exercise 2

Use the [`rand`](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand) function to generate a sample of Gaussian-distributed values. 

Plot a histogram of the resulting sample together with a Gaussian distribution that is normalized to the histogram.

## Exercise 3

Numerically show that the variance of a uniform distribution in the interval $[a,b]$ is $|b-a|^2/12$ using a sampling method of your choice. 

## Exercise 4

Use any random number generator to generate two arrays with 1000 entries that are 30% correlated.
