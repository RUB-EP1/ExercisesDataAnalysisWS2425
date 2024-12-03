# Tutorial 5: Wilks' theorem

## Wilks' theorem

Wilks' theorem is widely used in particle physics. It is important to know its implications and limitations to properly use it.<br>
Here are some points that help to guide the discussion

- What does the theorem state?
- How is it related to the central limit theorem?
- What is the big advantage of knowing the (asymptotic) distribution of a test statistic?
- How does the $\chi^2$ distribution relate to the normal distribution, and why is it important for likelihood ratio test statistics?
- What defines the number of degrees of freedom in the $\chi^2$ distribution?
- What are the limitations of Wilks' theorem? Would you apply it to Exercise 4?

<details> <summary> Further reading </summary>
In the context of Higgs, SUSY and other searches, Wilks' theorem has been extended to a broader range of test statistics,
and most importantly to distributions of the test statistic of alternative models: https://inspirehep.net/literature/860907

This allows to speed up the usage of the CL$_s$ method and others working with hypothesis test inversion,
as the toy-generation step can be skipped. It even applies to the profile-likelihood test statistic,
which would require fitting the generated toy data to compute the test statistic; and also this step can be skipped.
</details>

We will extend the codebase with the $\chi^2$ distribution, taking the following form:

````julia
"""
    χ²(x, n)

Compute the chi-squared probability density function at point `x` and degrees of freedom `n`.

# Arguments
- `x::Float64`: The value at which the density is evaluated.
- `n::Int`: The degrees of freedom.

# Returns
- `Float64`: The computed chi-squared density for the given `x` and `n`.

# Example
```julia
χ²(2.0, 3)
```
"""
````

<details> <summary> To copy and append to `test/runtests.jl`</summary>
Your `test/runtests.jl` file should look like this:

```julia
using Test
using DataAnalysisWS2425

include("test-distributions.jl")
include("test-sampling.jl")
include("test-fitting.jl")
include("test-plotting.jl")
include("test-utils.jl")
```

The following code should be appended to `test-fitting.jl`:

```julia
@testset "χ²" begin
    @test χ²(4.3,1) ≈ 0.0224100436236817
    @test χ²(2.0, 3) ≈ 0.20755374871029736
end
```

</details>
