# Tutorial 3: Tests for Fits

During this tutorial, you will copy over functions from Exercise sheet 2, and modify them to pass the tests that you can find on the bottom of this sheet.

> [!TIP]
> Reminder on how to run tests in [the setup instructions](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/setup.md#back-to-julia-running-tests).
> Also, to make the functions globally available, you need to export the functions in the src file: `export gaussian_scaled`.

## Functions

To pass the code tests for this tutorial (it will initially fail), the generic fitting function is needed.
This function calls the function that computes the extended negative log likelihood `extended_nll`, which will later be used for parameter estimation.
You can find the documentation that defines the function signatures below.

Coding the second function in the time of the tutorial is optional. However, it's a useful function to have to inspect your fit results and make pretty plots in your own favorite style.

### Function for computing extended negative log likelihood values

````julia
"""
    extended_nll(model, parameters, data; support = extrema(data), normalization_call = _quadgk_call)

Calculate the extended negative log likelihood (ENLL) for a given model and dataset.

# Arguments
- `model`: A function that represents the model. It should take two arguments: observable and parameters.
- `parameters`: The parameters for the model.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated. Defaults to the range of the data.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `extended_nll_value`: The extended negative log likelihood value for the given model and data.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
parameters = [1.0, 0.1]
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

enll_value = extended_nll(model, parameters, data; support=support)
```
"""
````

### Function for extended negative log likelihood fits

````julia
"""
    fit_enll(model, init_pars, data; support = extrema(data), alg=BFGS, normalization_call = _quadgk_call)

Fit the model parameters using the extended negative log likelihood (ENLL) method.

# Arguments
- `model`: A function that represents the model to be fitted. It should take two arguments: observable and parameters.
- `init_pars`: Initial parameters for the model.
- `data`: A collection of data points.
- `support`: (Optional) The domain over which the model is evaluated.
- `alg`: (Optional) Optimization algorithm to be used. Default is `BFGS`.
- `normalization_call`: (Optional) A function that calculates the normalization of the model over the support. Defaults to `_quadgk_call`.

# Returns
- `result`: The optimization result that minimizes the extended negative log likelihood.

# Example
```julia
model(x, p) = p[1] * exp(-p[2] * x)
init_pars = [1.0, 0.1]
data = [0.1, 0.2, 0.3, 0.4, 0.5]
support = (0.0, 1.0)

fit_result = fit_enll(model, init_pars, data; support=support)
```
"""
````

### Plotting function

````julia
"""
    plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlab="", ylab="")

Plot a histogram of data with a fit using model overlaid and a pull distribution.

# Arguments
- `data`: A collection of data points.
- `model`: A function that represents the model to be fitted. It should take two arguments: data points and parameters.
- `binning`: The bin edges for the histogram.
- `best_fit_pars`: The best-fit parameters for the model.
- `xlab`: (Optional) Label for the x-axis. Default is an empty string.
- `ylab`: (Optional) Label for the y-axis. Default is an empty string.

# Example
```julia
data = [0.1, 0.2, 0.3, 0.4, 0.5]
model(x, p) = p[1] * exp(-p[2] * x)
binning = 0:0.1:1.0
best_fit_pars = [1.0, 0.1]
plot_data_fit_with_pulls(data, model, binning, best_fit_pars; xlab="X-axis", ylab="Y-axis")
```
"""
````

<details> <summary> To copy and append to `test/runtests.jl`</summary>
Here is the code you copy over to your `test/runtests.jl` file

```julia
@testset "Simple fitting" begin
    init_pars = (; μ = 0.3, σ = 1.0, a = 1.0)
    support = (-4.0, 4.0)
    data = sample_inversion(400, support) do x
        gaussian_scaled(x; μ = 0.4, σ = 0.7, a = 1.0)
    end
    model(x, pars) = gaussian_scaled(x; pars.μ, pars.σ, pars.a)
    ext_unbinned_fit = fit_enll(model, init_pars, data; support)
    best_pars_extnll = typeof(init_pars)(ext_unbinned_fit.minimizer)
    @test ext_unbinned_fit.ls_success
    @test 0.36 < best_pars_extnll.μ < 0.44
end
```

</details>
