
"""
    interpolate_to_zero(two_x, two_y)

Interpolate to zero based on two points.

# Arguments
- `two_x::Vector`: A vector containing two x-values.
- `two_y::Vector`: A vector containing two y-values corresponding to `two_x`.

# Returns
- `Float64`: The interpolated x-value where y is zero.
"""
function interpolate_to_zero(two_x, two_y)
    w_left = 1 ./ two_y .* [1, -1]
    w_left ./= sum(w_left)
    return two_x' * w_left
end

"""
    find_zero_two_sides(xv, yv)

Find the zero crossings on both sides of the x-axis.

# Arguments
- `xv::AbstractVector`: A vector of x-values.
- `yv::AbstractVector`: A vector of y-values corresponding to `xv`.

# Returns
- `Vector{Float64}`: A vector containing two x-values where the y-values are zero.
"""
function find_zero_two_sides(xv, yv)
    yxv = yv .* xv
    _left = findfirst(x -> x > 0, yxv)
    _right = findlast(x -> x < 0, yxv)
    #
    x_left_zero = interpolate_to_zero([xv[_left-1], xv[_left]], [yv[_left-1], yv[_left]])
    x_right_zero =
        interpolate_to_zero([xv[_right], xv[_right+1]], [yv[_right], yv[_right+1]])
    #
    [x_left_zero, x_right_zero]
end
