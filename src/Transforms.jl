module Transforms
	using Interpolations
	using StaticArrays
	using LinearAlgebra
	using QuadGK
	using ProgressMeter
	
	export interpolate_array, line_segment_2D, LineFilterTransform, OrientationFilterTransform
	
	
	function interpolate_array(array::AbstractArray, interp_scheme::BSpline)
		interpolant = interpolate(array, interp_scheme)
		interpolant_padded = extrapolate(interpolant, 0)
	
		return interpolant_padded
	end
	
	function line_segment(i::Integer, j::Integer, θ::T, S::SVector{N, T}) where {N, T<:Real}
		
		x = SVector{N, T}(i+s*cos(θ) for s in S)
		y = SVector{N, T}(j+s*sin(θ) for s in S)
	
		return x, y
	end

	function line_segment(i::Integer, j::Integer, k::Integer, θ::T, ϕ::T, S::SVector{N, T}) where {N, T<:Real}
		
		x = SVector{N, T}(i+s*sin(θ)*cos(ϕ) for s in S)
		y = SVector{N, T}(j+s*sin(θ)*sin(ϕ) for s in S)
		z = SVector{N, T}(k+s*cos(θ) for s in S)
	
		return x, y, z
	end


	function LFT_point(i, j, Θ, array_interpolant, quad_points, quad_weights)
		
		function LFT_integrand(θ)
			line_seg = line_segment(i, j, θ, quad_points)
			array_vals = SVector([array_interpolant(c...) for c in zip(line_seg...)])
			return array_vals ⋅ quad_weights
		end
		
		LFT_θ = map(LFT_integrand, Θ)
		max_val, ind = findmax(LFT_θ)
		
		return max_val, Θ[ind]
	end

	function LFT_point(i, j, k, angles, array_interpolant, quad_points, quad_weights)
		
		function LFT_integrand(angle)
			line_seg = line_segment(i, j, k, angle..., quad_points)
			array_vals = SVector([array_interpolant(c...) for c in zip(line_seg...)])
			return array_vals ⋅ quad_weights
		end
		
		LFT_angles = map(LFT_integrand, angles)
		max_val, ind = findmax(LFT_angles)
		
		return max_val, angles[ind][1], angles[ind][2]
	end

	function LineFilterTransform(array::AbstractArray{T, 2}, N::Integer, r::Integer, Nr::Integer; interp_scheme::BSpline = BSpline(Constant())) where {T<:Real}
		array_interpolant = interpolate_array(array, interp_scheme)
		intensity, orientation = zero(array), zero(array)
		
		angles = SVector{N, T}(LinRange(-π/2, π/2, N))
		quad_points, quad_weights = gauss(Nr, -r, r)
		quad_points = SVector{Nr, T}(quad_points)
		quad_weights = SVector{Nr, T}(quad_weights)
		
		@showprogress for ind in CartesianIndices(array)
			intensity[ind], orientation[ind] = LFT_point(ind.I..., angles, array_interpolant, quad_points, quad_weights)
		end
	
		return intensity, orientation
	end

	function LineFilterTransform(array::AbstractArray{T, 3}, N::Integer, r::Integer, Nr::Integer; interp_scheme::BSpline = BSpline(Constant())) where {T<:Real}
		array_interpolant = interpolate_array(array, interp_scheme)
		intensity, orientation = zero(array), zeros(T, size(array)..., 2)
		
		angles = SVector{2N^2, Tuple{T, T}}(collect(Iterators.product(LinRange(-π/2, π/2, N), LinRange(-π, π, 2N))))
		quad_points, quad_weights = gauss(Nr, -r, r)
		quad_points = SVector{Nr, T}(quad_points)
		quad_weights = SVector{Nr, T}(quad_weights)
		
		@showprogress for ind in CartesianIndices(array)
			intensity[ind], orientation[ind,1], orientation[ind,2] = LFT_point(ind.I..., angles, array_interpolant, quad_points, quad_weights)
		end
	
		return intensity, orientation
	end
	
	function OFT_point(i, j, Α, intensity_field, orientation_field, quad_points, quad_weights)
		
		function OFT_integrand(α)
			line_seg = line_segment(i, j, α, quad_points)
			ρ_line = SVector([intensity_field(c...) for c in zip(line_seg...)])
			θ_line = SVector([orientation_field(c...) for c in zip(line_seg...)])
			
			orientation_functional = ρ_line.*cos.(2(θ_line .- α))
			return orientation_functional ⋅ quad_weights
		end
		
		OFT_α = map(OFT_integrand, Α)
		ind = argmax(abs.(OFT_α))
		
		return OFT_α[ind]
	end

	function OFT_point(i, j, k, angles, intensity_field, θ_field, ϕ_field, quad_points, quad_weights)
		
		function OFT_integrand(angle)
			line_seg = line_segment(i, j, k, angle..., quad_points)
			ρ_line = SVector([intensity_field(c...) for c in zip(line_seg...)])
			θ_line = SVector([θ_field(c...) for c in zip(line_seg...)])
			ϕ_line = SVector([ϕ_field(c...) for c in zip(line_seg...)])
			
			α₁, α₂ = angle
			inner = sin.(θ_line)*sin(α₁).*cos.(ϕ_line.-α₂).+cos.(θ_line)*cos(α₁)
			orientation_functional = ρ_line.*(2inner.^2 .- 1)
			return orientation_functional ⋅ quad_weights
		end
		
		OFT_angles = map(OFT_integrand, angles)
		ind = argmax(abs.(OFT_angles))
		
		return OFT_angles[ind]
	end

	function OrientationFilterTransform(LFT_intensity::AbstractArray{T, 2}, LFT_orientation::AbstractArray{T, 2}, N::Integer, r::Integer, Nr::Integer, interp_scheme::BSpline = BSpline(Constant())) where {T<:Real}
		ρ = interpolate_array(LFT_intensity, interp_scheme)
		θ = interpolate_array(LFT_orientation, interp_scheme)
		OFT = zero(LFT_intensity)
	
		angles = SVector{N, T}(LinRange(-π/2, π/2, N))
		quad_points, quad_weights = gauss(Nr, -r, r)
		quad_points = SVector{Nr, T}(quad_points)
		quad_weights = SVector{Nr, T}(quad_weights)
	
		@showprogress for ind in CartesianIndices(LFT_intensity)
			OFT[ind] = OFT_point(ind.I..., angles, ρ, θ, quad_points, quad_weights)
		end
	
		return OFT
	end

	function OrientationFilterTransform(LFT_intensity::AbstractArray{T, 3}, LFT_orientation::AbstractArray{T, 4}, N::Integer, r::Integer, Nr::Integer, interp_scheme::BSpline = BSpline(Constant())) where {T<:Real}
		ρ = interpolate_array(LFT_intensity, interp_scheme)
		θ = interpolate_array(LFT_orientation[:,:,:, 1], interp_scheme)
		ϕ = interpolate_array(LFT_orientation[:,:,:, 2], interp_scheme)
		OFT = zero(LFT_intensity)
	
		angles = SVector{2N^2, Tuple{T, T}}(collect(Iterators.product(LinRange(-π/2, π/2, N), LinRange(-π, π, 2N))))
		quad_points, quad_weights = gauss(Nr, -r, r)
		quad_points = SVector{Nr, T}(quad_points)
		quad_weights = SVector{Nr, T}(quad_weights)
	
		@showprogress for ind in CartesianIndices(LFT_intensity)
			OFT[ind] = OFT_point(ind.I..., angles, ρ, θ, ϕ, quad_points, quad_weights)
		end
	
		return OFT
	end
end