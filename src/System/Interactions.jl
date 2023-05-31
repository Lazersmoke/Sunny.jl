function empty_interactions(na, N)
    return map(1:na) do _
        Interactions(empty_anisotropy(N),
                     Coupling{Float64}[],
                     Coupling{Mat3}[],
                     Coupling{Float64}[])
    end
end

# Creates a clone of the lists of exchange interactions, which can be mutably
# updated.
function clone_interactions(ints::Interactions)
    (; aniso, heisen, exchange, biquad) = ints
    return Interactions(aniso, copy(heisen), copy(exchange), copy(biquad))
end

function interactions_homog(sys::System{N}) where N
    return sys.interactions_union :: Vector{Interactions}
end

function interactions_inhomog(sys::System{N}) where N
    return sys.interactions_union :: Array{Interactions, 4}
end

function is_homogeneous(sys::System{N}) where N
    return sys.interactions_union isa Vector{Interactions}
end

"""
    to_inhomogeneous(sys::System)

Returns a copy of the system that allows for inhomogeneous interactions, which
can be set using [`set_anisotropy_at!`](@ref), [`set_exchange_at!`](@ref),
[`set_biquadratic_at!`](@ref), and [`set_vacancy_at!`](@ref).

Inhomogeneous systems do not support symmetry-propagation of interactions or
system reshaping.
"""
function to_inhomogeneous(sys::System{N}) where N
    is_homogeneous(sys) || error("System is already inhomogeneous.")
    ints = interactions_homog(sys)

    ret = clone_system(sys)
    na = natoms(ret.crystal)
    ret.interactions_union = Array{Interactions}(undef, ret.latsize..., na)
    for i in 1:natoms(ret.crystal)
        for cell in all_cells(ret)
            ret.interactions_union[cell, i] = clone_interactions(ints[i])
        end
    end

    return ret
end


"""
    enable_dipole_dipole!(sys::System)

Enables long-range dipole-dipole interactions,

```math
    -(μ₀/4π) ∑_{⟨ij⟩}  (3 (𝐌_j⋅𝐫̂_{ij})(𝐌_i⋅𝐫̂_{ij}) - 𝐌_i⋅𝐌_j) / |𝐫_{ij}|^3
```

where the sum is over all pairs of spins (singly counted), including periodic
images, regularized using the Ewald summation convention. The magnetic moments
are ``𝐌_i = μ_B g 𝐒_i`` where ``g`` is the g-factor or g-tensor, and ``𝐒_i``
is the spin angular momentum dipole in units of ħ. The Bohr magneton ``μ_B`` and
vacuum permeability ``μ_0`` are physical constants, with numerical values
determined by the unit system.
"""
function enable_dipole_dipole!(sys::System{N}) where N
    sys.ewald = Ewald(sys)
    return
end

"""
    set_external_field!(sys::System, B::Vec3)

Sets the external field `B` that couples to all spins.
"""
function set_external_field!(sys::System, B)
    for site in all_sites(sys)
        set_external_field_at!(sys, B, site)
    end
end

"""
    set_external_field_at!(sys::System, B::Vec3, site::Site)

Sets a Zeeman coupling between a field `B` and a single spin. [`Site`](@ref)
includes a unit cell and a sublattice index.
"""
function set_external_field_at!(sys::System, B, site)
    sys.extfield[to_cartesian(site)] = Vec3(B)
end

"""
    set_vacancy_at!(sys::System, site::Site)

Make a single site nonmagnetic. [`Site`](@ref) includes a unit cell and a
sublattice index.
"""
function set_vacancy_at!(sys::System{N}, site) where N
    is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")

    site = to_cartesian(site)
    sys.κs[site] = 0.0
    sys.dipoles[site] = zero(Vec3)
    sys.coherents[site] = zero(CVec{N})
end


function local_energy_change(sys::System{N}, site, state::SpinState) where N
    (; s, Z) = state
    (; latsize, extfield, dipoles, coherents, ewald) = sys

    if is_homogeneous(sys)
        (; aniso, heisen, exchange, biquad) = interactions_homog(sys)[to_atom(site)]
    else
        (; aniso, heisen, exchange, biquad) = interactions_inhomog(sys)[site]
    end

    s₀ = dipoles[site]
    Z₀ = coherents[site]
    Δs = s - s₀
    ΔE = 0.0

    cell = to_cell(site)

    # Zeeman coupling to external field
    ΔE -= sys.units.μB * extfield[site] ⋅ (sys.gs[site] * Δs)

    # Single-ion anisotropy, dipole or SUN mode
    if N == 0
        E_new, _ = energy_and_gradient_for_classical_anisotropy(s, aniso.stvexp)
        E_old, _ = energy_and_gradient_for_classical_anisotropy(s₀, aniso.stvexp)
        ΔE += E_new - E_old
    else
        Λ = aniso.matrep
        ΔE += real(dot(Z, Λ, Z) - dot(Z₀, Λ, Z₀))
    end

    # Heisenberg exchange
    for (; bond, J) in heisen
        sⱼ = dipoles[offsetc(cell, bond.n, latsize), bond.j]
        ΔE += J * (Δs ⋅ sⱼ)    
    end

    # Quadratic exchange matrix
    for (; bond, J) in exchange
        sⱼ = dipoles[offsetc(cell, bond.n, latsize), bond.j]
        ΔE += dot(Δs, J, sⱼ)
    end

    # Scalar biquadratic exchange
    for (; bond, J) in biquad
        cellⱼ = offsetc(cell, bond.n, latsize)
        sⱼ = dipoles[cellⱼ, bond.j]
        if sys.mode == :dipole
            # Renormalization introduces a factor r and a Heisenberg term
            Sᵢ = (sys.Ns[site]-1)/2
            Sⱼ = (sys.Ns[cellⱼ, bond.j]-1)/2
            S = √(Sᵢ*Sⱼ)
            r = (1 - 1/S + 1/4S^2)
            ΔE += J * (r*((s⋅sⱼ)^2 - (s₀⋅sⱼ)^2) - (Δs⋅sⱼ)/2)
        elseif sys.mode == :large_S
            ΔE += J * ((s⋅sⱼ)^2 - (s₀⋅sⱼ)^2)
        elseif sys.mode == :SUN
            error("Biquadratic currently unsupported in SU(N) mode.") 
        end
    end

    # Long-range dipole-dipole
    if !isnothing(ewald)
        ΔE += ewald_energy_delta(sys, site, s)
    end

    return ΔE
end


"""
    energy(sys::System)

Computes the total system energy.
"""
function energy(sys::System{N}) where N
    (; crystal, dipoles, extfield, ewald) = sys

    E = 0.0

    # Zeeman coupling to external field
    for site in all_sites(sys)
        E -= sys.units.μB * extfield[site] ⋅ (sys.gs[site] * dipoles[site])
    end

    # Anisotropies and exchange interactions
    for i in 1:natoms(crystal)
        if is_homogeneous(sys)
            ints = interactions_homog(sys)
            E += energy_aux(sys, ints[i], i, all_cells(sys))
        else
            ints = interactions_inhomog(sys)
            for cell in all_cells(sys)
                E += energy_aux(sys, ints[cell, i], i, (cell, ))
            end
        end
    end

    # Long-range dipole-dipole
    if !isnothing(ewald)
        E += ewald_energy(sys)
    end
    
    return E
end

# Calculate the energy for the interactions `ints` defined for one sublattice
# `i` , accumulated over all equivalent `cells`.
function energy_aux(sys::System{N}, ints::Interactions, i::Int, cells) where N
    (; dipoles, coherents, latsize) = sys

    E = 0.0

    # Single-ion anisotropy
    if N == 0       # Dipole mode
        for cell in cells
            s = dipoles[cell, i]
            E += energy_and_gradient_for_classical_anisotropy(s, ints.aniso.stvexp)[1]
        end
    else            # SU(N) mode
        for cell in cells
            Λ = ints.aniso.matrep
            Z = coherents[cell, i]
            E += real(dot(Z, Λ, Z))
        end
    end

    # Heisenberg exchange
    for (; isculled, bond, J) in ints.heisen
        isculled && break
        for cell in cells
            sᵢ = dipoles[cell, bond.i]
            sⱼ = dipoles[offsetc(cell, bond.n, latsize), bond.j]
            E += J * dot(sᵢ, sⱼ)
        end
    end
    # Quadratic exchange matrix
    for (; isculled, bond, J) in ints.exchange
        isculled && break
        for cell in cells
            sᵢ = dipoles[cell, bond.i]
            sⱼ = dipoles[offsetc(cell, bond.n, latsize), bond.j]
            E += dot(sᵢ, J, sⱼ)
        end
    end
    # Scalar biquadratic exchange
    for (; isculled, bond, J) in ints.biquad
        isculled && break
        for cellᵢ in cells
            cellⱼ = offsetc(cellᵢ, bond.n, latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]
            if sys.mode == :dipole
                # Renormalization introduces a factor r and a Heisenberg term
                Sᵢ = (sys.Ns[cellᵢ, bond.i]-1)/2
                Sⱼ = (sys.Ns[cellⱼ, bond.j]-1)/2
                S = √(Sᵢ*Sⱼ)
                r = (1 - 1/S + 1/4S^2)
                E += J * (r*(sᵢ⋅sⱼ)^2 - (sᵢ⋅sⱼ)/2 + S^3 + S^2/4)
            elseif sys.mode == :large_S
                E += J * (sᵢ⋅sⱼ)^2
            elseif sys.mode == :SUN
                error("Biquadratic currently unsupported in SU(N) mode.")
            end
        end
    end

    return E
end


# Updates B in-place to hold negative energy gradient, -dE/ds, for each spin.
function set_forces!(B, dipoles::Array{Vec3, 4}, buf::Array{Vec3,4}, sys::System{N}) where N
    (; crystal, extfield, ewald) = sys

    fill!(B, zero(Vec3))

    # Zeeman coupling
    for site in all_sites(sys)
        B[site] += sys.units.μB * (sys.gs[site]' * extfield[site])
    end

    # Anisotropies and exchange interactions
    for i in 1:natoms(crystal)
        if is_homogeneous(sys)
            ints = interactions_homog(sys)
            #set_forces_aux!(B, dipoles, ints[i], i, all_cells(sys), sys)
            set_forces_aux_homog!(B, dipoles, buf, ints[i], i, sys)
        else
            ints = interactions_inhomog(sys)
            for cell in all_cells(sys)
                set_forces_aux!(B, dipoles, ints[cell, i], i, (cell, ), sys)
            end
        end
    end

    if !isnothing(ewald)
        accum_ewald_force!(B, dipoles, sys)
    end
end

# Calculate the energy for the interactions `ints` defined for one sublattice
# `i` , accumulated over all equivalent `cells`.
function set_forces_aux!(B, dipoles::Array{Vec3, 4}, ints::Interactions, i::Int, cells, sys::System{N}) where N
    (; latsize) = sys

    # Single-ion anisotropy only contributes in dipole mode. In SU(N) mode, the
    # anisotropy matrix will be incorporated directly into ℌ.
    if N == 0
        for cell in cells
            s = dipoles[cell, i]
            B[cell, i] -= energy_and_gradient_for_classical_anisotropy(s, ints.aniso.stvexp)[2]
        end
    end

    # Heisenberg exchange
    for (; isculled, bond, J) in ints.heisen
        isculled && break
        for cellᵢ in cells
            cellⱼ = offsetc(cellᵢ, bond.n, latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]
            B[cellᵢ, bond.i] -= J  * sⱼ
            B[cellⱼ, bond.j] -= J' * sᵢ
        end
    end
    # Quadratic exchange matrix
    for (; isculled, bond, J) in ints.exchange
        isculled && break
        for cellᵢ in cells
            cellⱼ = offsetc(cellᵢ, bond.n, latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]
            B[cellᵢ, bond.i] -= J  * sⱼ
            B[cellⱼ, bond.j] -= J' * sᵢ
        end
    end
    # Scalar biquadratic exchange
    for (; isculled, bond, J) in ints.biquad
        isculled && break
        for cellᵢ in cells
            cellⱼ = offsetc(cellᵢ, bond.n, latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]

            if sys.mode == :dipole
                Sᵢ = (sys.Ns[cellᵢ, bond.i]-1)/2
                Sⱼ = (sys.Ns[cellⱼ, bond.j]-1)/2
                S = √(Sᵢ*Sⱼ)
                # Renormalization introduces a factor r and a Heisenberg term
                r = (1 - 1/S + 1/4S^2)
                B[cellᵢ, bond.i] -= J * (2r*sⱼ*(sᵢ⋅sⱼ) - sⱼ/2)
                B[cellⱼ, bond.j] -= J * (2r*sᵢ*(sᵢ⋅sⱼ) - sᵢ/2)
            elseif sys.mode == :large_S
                B[cellᵢ, bond.i] -= J * 2sⱼ*(sᵢ⋅sⱼ)
                B[cellⱼ, bond.j] -= J * 2sᵢ*(sᵢ⋅sⱼ)
            elseif sys.mode == :SUN
                error("Biquadratic currently unsupported in SU(N) mode.")
            end
        end
    end
end

function set_forces_aux_homog!(B, dipoles::Array{Vec3, 4}, buf::Array{Vec3,4}, ints::Interactions, i::Int, sys::System{N}) where N
    (; latsize) = sys

    cells = all_cells(sys);

    subLatBuffer = view(buf,:,:,:,1);

    # Single-ion anisotropy only contributes in dipole mode. In SU(N) mode, the
    # anisotropy matrix will be incorporated directly into ℌ.
    if N == 0

        for cell in cells
            s = dipoles[cell, i]
            B[cell, i] -= energy_and_gradient_for_classical_anisotropy(s, ints.aniso.stvexp)[2]
        end
        #f = s -> energy_and_gradient_for_classical_anisotropy(s,ints.aniso.stvexp)[2]
        #@. xx = f(dipoles)
        #@. B -= xx
    end


    # Heisenberg exchange
    for (; isculled, bond, J) in ints.heisen
        isculled && break
        # Forwards interaction: j there contributes to i here
        fastcircshift3!(subLatBuffer,view(dipoles,:,:,:,bond.j),-bond.n)
        subLatBuffer .*= J
        @views B[:,:,:,bond.i] .-= subLatBuffer

        # Backwards interaction: i here contributes to j there
        fastcircshift3!(subLatBuffer,view(dipoles,:,:,:,bond.i),+bond.n)
        subLatBuffer .*= J'
        @views B[:,:,:,bond.j] .-= subLatBuffer
    end
    # Quadratic exchange matrix
    for (; isculled, bond, J) in ints.exchange
        isculled && break
        for cellᵢ in cells
            cellⱼ = offsetc(cellᵢ, bond.n, latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]
            B[cellᵢ, bond.i] -= J  * sⱼ
            B[cellⱼ, bond.j] -= J' * sᵢ
        end
    end
    # Scalar biquadratic exchange
    for (; isculled, bond, J) in ints.biquad
        isculled && break
        for cellᵢ in cells
            cellⱼ = offsetc(cellᵢ, bond.n, latsize)
            sᵢ = dipoles[cellᵢ, bond.i]
            sⱼ = dipoles[cellⱼ, bond.j]

            if sys.mode == :dipole
                Sᵢ = (sys.Ns[cellᵢ, bond.i]-1)/2
                Sⱼ = (sys.Ns[cellⱼ, bond.j]-1)/2
                S = √(Sᵢ*Sⱼ)
                # Renormalization introduces a factor r and a Heisenberg term
                r = (1 - 1/S + 1/4S^2)
                B[cellᵢ, bond.i] -= J * (2r*sⱼ*(sᵢ⋅sⱼ) - sⱼ/2)
                B[cellⱼ, bond.j] -= J * (2r*sᵢ*(sᵢ⋅sⱼ) - sᵢ/2)
            elseif sys.mode == :large_S
                B[cellᵢ, bond.i] -= J * 2sⱼ*(sᵢ⋅sⱼ)
                B[cellⱼ, bond.j] -= J * 2sᵢ*(sᵢ⋅sⱼ)
            elseif sys.mode == :SUN
                error("Biquadratic currently unsupported in SU(N) mode.")
            end
        end
    end
end

function fastcircshift3!(dest,src,shift)
  s = size(src)
  ix = mod1(-shift[1], s[1])
  iy = mod1(-shift[2], s[2])
  iz = mod1(-shift[3], s[3])

  idx = s[1] - ix
  idy = s[2] - iy
  idz = s[3] - iz

  # 8 blocks to copy
  # dx,dy,dz
  #dest[1:idx,1:idy,1:idz] .= view(src,(ix+1):s[1],(iy+1):s[2],(iz+1):s[3])
  copyto!(dest,CartesianIndices((1:idx,1:idy,1:idz)),src,CartesianIndices(((ix+1):s[1],(iy+1):s[2],(iz+1):s[3])))
  # dx,dy,z
  copyto!(dest,CartesianIndices((1:idx,1:idy,(idz+1):s[3])),src,CartesianIndices(((ix+1):s[1],(iy+1):s[2],1:iz)))
  # dx,y,dz
  copyto!(dest,CartesianIndices((1:idx,(idy+1):s[2],1:idz)),src,CartesianIndices(((ix+1):s[1],1:iy,(iz+1):s[3])))
  # dx,y,z
  copyto!(dest,CartesianIndices((1:idx,(idy+1):s[2],(idz+1):s[3])),src,CartesianIndices(((ix+1):s[1],1:iy,1:iz)))
  # x,dy,dz
  copyto!(dest,CartesianIndices(((idx+1):s[1],1:idy,1:idz)),src,CartesianIndices((1:ix,(iy+1):s[2],(iz+1):s[3])))
  # x,dy,z
  copyto!(dest,CartesianIndices(((idx+1):s[1],1:idy,(idz+1):s[3])),src,CartesianIndices((1:ix,(iy+1):s[2],1:iz)))
  # x,y,dz
  copyto!(dest,CartesianIndices(((idx+1):s[1],(idy+1):s[2],1:idz)),src,CartesianIndices((1:ix,1:iy,(iz+1):s[3])))
  # x,y,z
  copyto!(dest,CartesianIndices(((idx+1):s[1],(idy+1):s[2],(idz+1):s[3])),src,CartesianIndices((1:ix,1:iy,1:iz)))
  return nothing
end

"""
    forces(Array{Vec3}, sys::System)

Returns the effective local field (force) at each site, ``𝐁 = -∂E/∂𝐬``.
"""
function forces(sys::System{N}) where N
    B = zero(sys.dipoles)
    set_forces!(B, sys.dipoles, sys)
    return B
end
