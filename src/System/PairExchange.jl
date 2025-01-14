function validate_bond(cryst::Crystal, bond::Bond)
    # Verify bond indices
    if bond.i == bond.j && iszero(bond.n)
        error("Bond must connect different atoms.")
    end
    (1 <= bond.i <= natoms(cryst)) || error("Atom index $(bond.i) is out of range.")
    (1 <= bond.j <= natoms(cryst)) || error("Atom index $(bond.j) is out of range.")
end

# Partition every nonzero bound into one of two sets
function bond_parity(bond)
    bond_delta = (bond.j - bond.i, bond.n...)
    @assert bond_delta != (0, 0, 0, 0)
    return bond_delta > (0, 0, 0, 0)
end


"""
    set_biquadratic!(sys::System, J, bond::Bond)

Sets a scalar biquadratic interaction along `bond`, yielding a pairwise energy
``J (𝐒_i⋅𝐒_j)²``. This interaction will be propagated to equivalent bonds in
consistency with crystal symmetry. Any previous biquadratic exchange
interactions on these bonds will be overwritten.

For systems restricted to dipoles, the biquadratic interactions will
automatically be renormalized to achieve maximum consistency with the more
variationally accurate SU(_N_) mode. This renormalization introduces also a
correction to the quadratic part of the exchange.

See also [`set_exchange!`](@ref).
"""
function set_biquadratic!(sys::System{N}, J, bond::Bond) where N
    is_homogeneous(sys) || error("Use `set_biquadratic_at!` for an inhomogeneous system.")

    # If `sys` has been reshaped, then operate first on `sys.origin`, which
    # contains full symmetry information.
    if !isnothing(sys.origin)
        set_biquadratic!(sys.origin, J, bond)
        set_interactions_from_origin!(sys)
        return
    end

    validate_bond(sys.crystal, bond)

    ints = interactions_homog(sys)


    # Print a warning if an interaction already exists for bond
    if any(x -> x.bond == bond, ints[bond.i].biquad)
        println("Warning: Overriding biquadratic interaction for bond $bond.")
    end

    for i in 1:natoms(sys.crystal)
        bonds = all_symmetry_related_bonds_for_atom(sys.crystal, i, bond)
        isempty(bonds) && continue

        for bond′ in bonds
            # Remove any existing exchange for bond′
            matches_bond(c) = c.bond == bond′
            filter!(!matches_bond, ints[i].biquad)

            # The energy or force calculation only needs to see each bond once
            isculled = bond_parity(bond′)

            # Add to list
            coupling = Coupling(isculled, bond′, J)
            push!(ints[i].biquad, coupling)
        end

        # Sort interactions so that non-culled bonds appear first
        sort!(ints[i].biquad, by=c->c.isculled)
    end
end


"""
    set_exchange!(sys::System, J, bond::Bond)

Sets a 3×3 spin-exchange matrix `J` along `bond`, yielding a pairwise
interaction energy ``𝐒_i⋅J 𝐒_j``. This interaction will be propagated to
equivalent bonds in consistency with crystal symmetry. Any previous exchange
interactions on these bonds will be overwritten. The parameter `bond` has the
form `Bond(i, j, offset)`, where `i` and `j` are atom indices within the unit
cell, and `offset` is a displacement in unit cells.

Scalar `J` implies a pure Heisenberg exchange.

As a convenience, `dmvec(D)` can be used to construct the antisymmetric part of
the exchange, where `D` is the Dzyaloshinskii-Moriya pseudo-vector. The
resulting interaction will be ``𝐃⋅(𝐒_i×𝐒_j)``.

# Examples
```julia
using Sunny, LinearAlgebra

# An explicit exchange matrix
J1 = [2 3 0;
     -3 2 0;
      0 0 2]
set_exchange!(sys, J1, bond)

# An equivalent Heisenberg + DM exchange 
J2 = 2*I + dmvec([0,0,3])
set_exchange!(sys, J2, bond)
```

See also [`set_biquadratic!`](@ref), [`dmvec`](@ref).
"""
function set_exchange!(sys::System{N}, J, bond::Bond) where N
    is_homogeneous(sys) || error("Use `set_exchange_at!` for an inhomogeneous system.")

    # If `sys` has been reshaped, then operate first on `sys.origin`, which
    # contains full symmetry information.
    if !isnothing(sys.origin)
        set_exchange!(sys.origin, J, bond)
        set_interactions_from_origin!(sys)
        return
    end

    validate_bond(sys.crystal, bond)
    ints = interactions_homog(sys)

    # Convert J to Mat3
    J = Mat3(J isa Number ? J*I : J)
    is_heisenberg = isapprox(diagm([J[1,1],J[1,1],J[1,1]]), J; atol=1e-12)

    # Verify that exchange is symmetry-consistent
    if !is_coupling_valid(sys.crystal, bond, J)
        println("Symmetry-violating exchange: $J.")
        println("Use `print_bond(crystal, $bond)` for more information.")
        error("Interaction violates symmetry.")
    end

    # Print a warning if an interaction already exists for bond
    if any(x -> x.bond == bond, vcat(ints[bond.i].heisen, ints[bond.i].exchange))
        println("Warning: Overriding exchange for bond $bond.")
    end

    for i in 1:natoms(sys.crystal)
        bonds, Js = all_symmetry_related_couplings_for_atom(sys.crystal, i, bond, J)
        isempty(bonds) && continue

        for (bond′, J′) in zip(bonds, Js)
            # Remove any existing exchange for bond′
            matches_bond(c) = c.bond == bond′
            filter!(!matches_bond, ints[i].heisen)
            filter!(!matches_bond, ints[i].exchange)

            # The energy or force calculation only needs to see each bond once
            isculled = bond_parity(bond′)

            # Add to list
            if is_heisenberg
                coupling = Coupling(isculled, bond′, J′[1,1])
                push!(ints[i].heisen, coupling)
            else
                coupling = Coupling(isculled, bond′, J′)
                push!(ints[i].exchange, coupling)
            end
        end

        # Sort interactions so that non-culled bonds appear first
        sort!(ints[i].heisen, by=c->c.isculled)
        sort!(ints[i].exchange, by=c->c.isculled)
    end
end

# Converts two sites to a bond with indices for a reshaped system. For internal
# use only.
function sites_to_internal_bond(site1::CartesianIndex{4}, site2::CartesianIndex{4})
    n = Tuple(to_cell(site2)) .- Tuple(to_cell(site1))
    return Bond(to_atom(site1), to_atom(site2), n)
end

# Internal function only
function push_coupling!(couplings, bond, J)
    isculled = bond_parity(bond)
    filter!(c -> c.bond != bond, couplings)
    push!(couplings, Coupling(isculled, bond, J))
    sort!(couplings, by=c->c.isculled)
    return
end

"""
    set_biquadratic_at!(sys::System, J, site1::Site, site2::Site)

Sets the scalar biquadratic interaction along the single bond connecting two
[`Site`](@ref)s, ignoring crystal symmetry. The system must support
inhomogeneous interactions via [`to_inhomogeneous`](@ref).

See also [`set_biquadratic!`](@ref).
"""
function set_biquadratic_at!(sys::System{N}, J, site1, site2) where N
    site1 = to_cartesian(site1)
    site2 = to_cartesian(site2)
    bond = sites_to_internal_bond(site1, site2)

    is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")
    ints = interactions_inhomog(sys)

    push_coupling!(ints[site1].biquad, bond, J)
    push_coupling!(ints[site2].biquad, reverse(bond), J')
    return
end


"""
    set_exchange_at!(sys::System, J, site1::Site, site2::Site)

Sets the exchange interaction along the single bond connecting two
[`Site`](@ref)s, ignoring crystal symmetry. The system must support
inhomogeneous interactions via [`to_inhomogeneous`](@ref).

See also [`set_exchange!`](@ref).
"""
function set_exchange_at!(sys::System{N}, J, site1, site2) where N
    site1 = to_cartesian(site1)
    site2 = to_cartesian(site2)
    bond = sites_to_internal_bond(site1, site2)

    is_homogeneous(sys) && error("Use `to_inhomogeneous` first.")
    ints = interactions_inhomog(sys)

    # Convert J to Mat3
    J = Mat3(J isa Number ? J*I : J)
    is_heisenberg = isapprox(diagm([J[1,1],J[1,1],J[1,1]]), J; atol=1e-12)
    
    if is_heisenberg
        push_coupling!(ints[site1].heisen, bond, J[1,1])
        push_coupling!(ints[site2].heisen, reverse(bond), J[1,1]')
    else
        push_coupling!(ints[site1].exchange, bond, J)
        push_coupling!(ints[site2].exchange, reverse(bond), J')
    end
    return
end


"""
    dmvec(D)

Antisymmetric matrix representation of the Dzyaloshinskii-Moriya pseudo-vector,

```
  [  0    D[3] -D[2]
   -D[3]   0    D[1]
    D[2] -D[1]   0  ]
```

Useful in the context of [`set_exchange!`](@ref).
"""
function dmvec(D)
    D = Vec3(D)
    return SA[  0.0  D[3] -D[2]
              -D[3]   0.0  D[1]
               D[2] -D[1]   0.0 ]
end
