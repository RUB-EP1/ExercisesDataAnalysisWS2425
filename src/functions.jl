
"""
    gaussian_scaled(x; μ, σ, a=1.0)

Computes the value of a Gaussian function with flexible normalization at `x`, given the mean `μ`, standard deviation `σ`, and scaling factor `a`.

The form of the Gaussian is:

    a * exp(-((x - μ)^2) / (2 * σ^2))

# Example
```julia
julia> y = gaussian_scaled(2.0; μ=0.0, σ=1.0, a=3.0)
```
"""
gaussian_scaled(x; μ, σ, a) = a * exp(-(x - μ)^2 / (2σ^2))

"""
    polynomial_scaled(x; coeffs)

Evaluates a polynomial at `x`, given the coefficients in `coeffs`.
The `coeffs` is an iterable collection of coefficients, where the first element corresponds to the lowest degree term.

The polynomial has the form:

    coeffs[1] * x^0 + coeffs[1] * x^(2) + ... + coeffs[n] * x^(n-1)

where `n-1` is the degree of the polynomial, determined by the length of `coeffs`.

# Example
```julia
julia> y = polynomial_scaled(2.0; coeffs=[1.0, -3.0, 2.0])
```
"""
function polynomial_scaled(x; coeffs)
    result = 0
    for (i, coeff) in enumerate(coeffs)
        result += coeff * x^(i - 1)
    end
    return result
end


"""
    breit_wigner_scaled(x; M, Γ, a=1.0)

Computes the value of a Breit-Wigner function with flexible normalization at `x`, given the mass `m`, width `Γ`, and scaling factor `a`.

The form of the Breit-Wigner is:

    a / |m^2 - x^2 - imΓ|^2

# Example
```julia
julia> y = breit_wigner_scaled(2.0; m=1.0, Γ=0.5, a=2.0)
```
"""
breit_wigner_scaled(x; M, Γ, a) = a / abs2(M^2 - x^2 - 1im * M * Γ)

"""
    voigt_scaled(x; m=0.0, Γ=1.0, σ=1.0, a=1.0)

Computes the value of a Voigt profile with flexible normalization at `x`, given the peak position `m`, Breit-Wigner width `Γ`, Gaussian width `σ`, and scaling factor `a`.

The Voigt profile is a convolution of a non-relativistic Breit-Wigner function and a Gaussian, commonly used to describe spectral lineshapes.

# Example
```julia
julia> y = voigt_scaled(2.0; m=1.3, Γ=0.15, σ=0.3, a=3.0)
```
"""
voigt_scaled(x; M, Γ, σ, a) = quadgk(-Inf, Inf) do τ
    breit_wigner_scaled(x - τ; M, Γ, a) * gaussian_scaled(τ; μ = 0, σ, a)
end[1]
