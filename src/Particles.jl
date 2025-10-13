# compute characteristic time for subgrid diffusion
@parallel_indices (I...) function subgrid_characteristic_time!(dt₀, ρCp, λ, dxi)

    di = getindex.(dxi, I...)
    # Compute the characteristic timescale `dt₀` of the local cell
    ρCpᵢ = ρCp[I...]
    sum_dxi = mapreduce(x -> inv(x)^2, +, di)
    dt₀[I...] = ρCpᵢ / (2 * λ * sum_dxi)

    return nothing
end

# using Adapt
struct StressParticles{backend, nNormal, nShear, T}
    τ_normal::NTuple{nNormal, T}
    τ_shear::NTuple{nShear, T}
    ω::NTuple{nShear, T}

    function StressParticles(
            backend, τ_normal::NTuple{nNormal, T}, τ_shear::NTuple{nShear, T}, ω::NTuple{nShear, T}
        ) where {nNormal, nShear, T}
        return new{backend, nNormal, nShear, T}(τ_normal, τ_shear, ω)
    end
end

function StressParticles(particles::Particles{backend, 2}) where {backend}
    τ_normal = init_cell_arrays(particles, Val(2)) # normal stress
    τ_shear = init_cell_arrays(particles, Val(1)) # normal stress
    ω = init_cell_arrays(particles, Val(1)) # vorticity

    return StressParticles(backend, τ_normal, τ_shear, ω)
end

@inline unwrap(x::StressParticles) = tuple(x.τ_normal..., x.τ_shear..., x.ω...)
@inline normal_stress(x::StressParticles) = x.τ_normal
@inline shear_stress(x::StressParticles) = x.τ_shear
@inline shear_vorticity(x::StressParticles) = x.ω


# Vorticity tensor

@parallel function compute_vorticity!(ωxy, Vx, Vy, dxc_I, dyc_I)

    @all(ωxy) = 0.5 * (@d_xa(Vy) * @all(dxc_I) - @d_ya(Vx) * @all(dyc_I))

    return nothing
end


## Stress Rotation on the particles

function rotate_stress_particles!(
        τ::NTuple, ω::NTuple, particles::Particles, dt
    )

    nx, ny = size(particles.index)
    @parallel (1:nx, 1:ny) rotate_stress_particles_GeoParams!(
        τ..., ω..., particles.index, dt
    )

    return nothing
end

@parallel_indices (I...) function rotate_stress_particles_GeoParams!(
        xx, yy, xy, ω, index, dt
    )
    for ip in cellaxes(index)
        @index(index[ip, I...]) || continue # no particle in this location

        ω_xy = @inbounds @index ω[ip, I...]
        τ_xx = @inbounds @index xx[ip, I...]
        τ_yy = @inbounds @index yy[ip, I...]
        τ_xy = @inbounds @index xy[ip, I...]

        τ_rotated = GeoParams.rotate_elastic_stress2D(ω_xy, (τ_xx, τ_yy, τ_xy), dt)

        @inbounds @index xx[ip, I...] = τ_rotated[1]
        @inbounds @index yy[ip, I...] = τ_rotated[2]
        @inbounds @index xy[ip, I...] = τ_rotated[3]
    end

    return nothing
end

# Interpolations between stress on the particles and the grid

function stress2grid!(
        stokes, τ_particles::StressParticles{backend}, xvi, xci, particles
    ) where {backend}
    return stress2grid!(
        stokes,
        normal_stress(τ_particles)...,
        shear_stress(τ_particles)...,
        xvi,
        xci,
        particles,
    )
end

function stress2grid!(τxx_o, τyy_o, τxy_o, pτ, xvi, xci, particles)
    # normal components
    particle2centroid!(τxx_o, pτ.τ_normal[1], xci, particles)
    particle2centroid!(τyy_o, pτ.τ_normal[2], xci, particles)
    # shear components
    particle2grid!(τxy_o, pτ.τ_shear[1], xvi, particles)
    return nothing
end


function rotate_stress!(pτ, τxx_o, τyy_o, τxy_o, ω_xy, particles, xci, xvi, dt)
    # normal components
    centroid2particle!(pτ.τ_normal[1], xci, τxx_o, particles)
    centroid2particle!(pτ.τ_normal[2], xci, τyy_o, particles)
    # shear components
    grid2particle!(pτ.τ_shear[1], xvi, τxy_o, particles)
    # vorticity tensor
    grid2particle!(pτ.ω[1], xvi, ω_xy, particles)
    # rotate stress
    rotate_stress_particles!((pτ.τ_normal..., pτ.τ_shear...), pτ.ω, particles, dt)

    return nothing
end