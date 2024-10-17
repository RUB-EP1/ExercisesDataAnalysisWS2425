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
b) $\lambda=0.05$<br>
c) $a=0.01, b=-0.04, c=0.25, d=15$ in the range [-10,10]<br>
d) $M=1530, \Gamma=9$<br>
e) $M=1530, \Gamma=9, \sigma=16$

Make sure that the distributions are probability density functions (in the given range)

> [!TIP]
> Read about [Plots](https://docs.juliaplots.org/latest/) in the julia `Plots` package manual.

## Exercise 2

Use the [`rand`](https://docs.julialang.org/en/v1/stdlib/Random/#Base.rand) function and rejection sampling to generate a sample of Gaussian-distributed values.

Plot a histogram of the resulting sample together with a Gaussian distribution that is normalized to the histogram.

## Exercise 3

Numerically show that the variance of a uniform distribution in the interval $[a,b]$ is $|b-a|^2/12$ using a sampling method of your choice.

## Exercise 4

Use any random number generator to generate two arrays with 1000 entries that are 30% correlated.


## Test at tutorial

The functions that needs to be propagated to the code base.
We would like to define
- name of the function
- signature
- add tests to check that the imoplementation is connect

1. Gaussian function

signature: `gaussian_unnorm(x; \mu, \sigma, a)`

```julia
@test gaussian_unnorm(1.1; ...) = 2.2
@test gaussian_unnorm(1.1; other_pars = 2.3
```

2. Polynomial

signature: `polynomial_unnorm(x; coeff::Tuple)`

```julia
@test polynomial_unnorm(1.1; coeff=(1.1)) = 1.1
@test polynomial_unnorm(3.1; coeff=(1.1)) = 1.1
@test polynomial_unnorm(1.1; coeff=(11.1, 22.2)) = 2.2
@test polynomial_unnorm(1.1; coeff=(33.1, 22.2, 2.4)) = 3.3
```

3. BW

signature: `voigt_unnorm(x; a, \mu, \Gamma)`

```julia
@test voigt_unnorm(1.1; ...) = 1.1
@test voigt_unnorm(3.1; ...) = 1.1
```

4. Voigt

signature: `voigt_unnorm(x; a, \mu, \sigma, \Gamma)`

```julia
@test voigt_unnorm(1.1; ...) = 1.1
@test voigt_unnorm(3.1; ...) = 1.1
```

5. Sampling with H&M

signature: `sample_hit_and_miss(function_to_sample; trial_number::Int, support::Tuple)`

```julia
Random.seed!(4321)
@test sample_hit_and_miss(f; ...) = [3.3, 2.3, 3.2]
@test sample_hit_and_miss(f; ...) = [3.3, 3.2]
```

6. Sampling with inversion

signature: `sample_inverse(function_to_sample; target_number::Int, support::Tuple, n_grid::Int)`

```julia
Random.seed!(4321)
@test sample_inverse(f; ...) = [3.3, 2.3, 3.2]
@test sample_inverse(f; ...) = [3.3, 3.2]
```

7. Correlation

signature: `correlation(x_array, y_array)`

```julia
Random.seed!(4321)
xv = rand(3)
yv = randn(3)
@test correlation(xv,yv) = 0.23
@test correlation(xv,2*yv) = correlation(xv,yv)
@test correlation(xv,xv+yv) = 0.44
```