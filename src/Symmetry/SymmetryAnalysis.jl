# Wrap each coordinate of position r into the range [0,1). To account for finite
# precision, wrap 1-ϵ to -ϵ, where ϵ=symprec is a tolerance parameter.
function wrap_to_unit_cell(r::Vec3; symprec)
    return @. mod(r+symprec, 1) - symprec
end

function all_integer(x; symprec)
    return norm(x - round.(x)) < symprec
end

function is_periodic_copy(cryst::Crystal, r1::Vec3, r2::Vec3)
    all_integer(r1-r2; cryst.symprec)
end

function is_periodic_copy(cryst::Crystal, b1::BondPos, b2::BondPos)
    # Displacements between the two bonds
    D1 = b2.ri - b1.ri
    D2 = b2.rj - b1.rj
    # Round components of D1 to nearest integers
    n = round.(D1, RoundNearest)
    # If both n ≈ D1 and n ≈ D2, then the bonds are equivalent by translation
    return norm(n - D1) < cryst.symprec && norm(n - D2) < cryst.symprec
end

function position_to_atom(cryst::Crystal, r::Vec3)
    return findfirst(r′ -> is_periodic_copy(cryst, r, r′), cryst.positions)
end

function position_to_atom_and_offset(cryst::Crystal, r::Vec3)
    i = position_to_atom(cryst, r)::Int
    # See comment in wrap_to_unit_cell() regarding shift by symprec
    offset = @. round(Int, r+cryst.symprec, RoundDown)
    @assert isapprox(cryst.positions[i]+offset, r; atol=cryst.symprec)
    return (i, offset)
end


# Generate list of SymOps for the pointgroup of atom i
function symmetries_for_pointgroup_of_atom(cryst::Crystal, i::Int)
    ret = SymOp[]
    r = cryst.positions[i]
    for s in cryst.symops
        r′ = transform(s, r)
        if is_periodic_copy(cryst, r, r′)
            push!(ret, s)
        end
    end
    return ret
end


# General a list of all symmetries that transform i2 into i1. (Convention for
# definition of `s` is consistent with symmetries_between_bonds())
function symmetries_between_atoms(cryst::Crystal, i1::Int, i2::Int)
    ret = SymOp[]
    r1 = cryst.positions[i1]
    r2 = cryst.positions[i2]
    for s in cryst.symops
        if is_periodic_copy(cryst, r1, transform(s, r2))
            push!(ret, s)
        end
    end
    return ret
end


# The list of atoms symmetry-equivalent to i_ref
function all_symmetry_related_atoms(cryst::Crystal, i_ref::Int)
    # The result is the set of atoms sharing the symmetry "class"
    c = cryst.classes[i_ref]
    ret = findall(==(c), cryst.classes)

    # Calculate the result another way, as a consistency check.
    equiv_atoms = Int[]
    r_ref = cryst.positions[i_ref]
    for s in cryst.symops
        push!(equiv_atoms, position_to_atom(cryst, transform(s, r_ref)))
    end
    @assert sort(unique(equiv_atoms)) == ret

    return ret
end


# For each atom in the unit cell of `cryst`, return the corresponding element of
# `ref_atom` that is symmetry equivalent. Print a helpful error message if two
# reference atoms are symmetry equivalent, or if a reference atom is missing. 
function propagate_reference_atoms(cryst::Crystal, ref_atoms::Vector{Int})
    # Sort infos by site equivalence class
    ref_atoms = sort(ref_atoms; by = (a -> cryst.classes[a]))
    ref_classes = cryst.classes[ref_atoms]

    # Verify that none of the atoms belong to the same class
    for i = 1:length(ref_atoms)-1
        a1, a2 = ref_atoms[[i,i+1]]
        c1, c2 = ref_classes[[i,i+1]]
        if c1 == c2
            error("Atoms $a1 and $a2 are symmetry equivalent.")
        end
    end
    @assert allunique(ref_classes)

    # Verify that every class has been specified
    missing_classes = setdiff(cryst.classes, ref_classes)
    if !isempty(missing_classes)
        c = first(missing_classes)
        a = findfirst(==(c), cryst.classes)
        error("Not all sites are specified; consider including atom $a.")
    end
    @assert length(ref_atoms) == length(unique(cryst.classes))

    # Return a symmetry-equivalent reference atom for each atom in the unit cell
    return map(cryst.classes) do c
        ref_atoms[only(findall(==(c), ref_classes))]
    end
end


# Generate list of all symmetries that transform b2 into b1, along with parity
function symmetries_between_bonds(cryst::Crystal, b1::BondPos, b2::BondPos)
    # Fail early if two bonds describe different real-space distances
    # (dimensionless error tolerance is measured relative to the minimum lattice
    # constant ℓ)
    if b1 != b2
        ℓ = minimum(norm, eachcol(cryst.latvecs))
        d1 = global_distance(cryst, b1) / ℓ
        d2 = global_distance(cryst, b2) / ℓ
        if abs(d1-d2) > cryst.symprec
            return Tuple{SymOp, Bool}[]
        end
    end

    ret = Tuple{SymOp, Bool}[]
    for s in cryst.symops
        b2′ = transform(s, b2)
        if is_periodic_copy(cryst, b1, b2′)
            push!(ret, (s, true))
        elseif is_periodic_copy(cryst, b1, reverse(b2′))
            push!(ret, (s, false))
        end
    end
    return ret
end

# Is there a symmetry operation that transforms `b1` into either `b2` or its
# reverse?
function is_related_by_symmetry(cryst::Crystal, b1::BondPos, b2::BondPos)
    return !isempty(symmetries_between_bonds(cryst::Crystal, b1::BondPos, b2::BondPos))
end

function is_related_by_symmetry(cryst::Crystal, b1::Bond, b2::Bond)
    return is_related_by_symmetry(cryst, BondPos(cryst, b1), BondPos(cryst, b2))
end

# Returns all bonds in `cryst` for which `bond.i == i`
function all_bonds_for_atom(cryst::Crystal, i::Int, max_dist; min_dist=0.0)
    # be a little generous with the minimum and maximum distances
    ℓ = minimum(norm, eachcol(cryst.latvecs))
    max_dist += 4 * cryst.symprec * ℓ
    min_dist -= 4 * cryst.symprec * ℓ

    # columns are the reciprocal vectors
    recip_vecs = 2π * inv(cryst.latvecs)'

    # box_lengths[i] represents the perpendicular distance between two parallel
    # boundary planes spanned by lattice vectors a_j and a_k (where indices j
    # and k differ from i)
    box_lengths = [a⋅b/norm(b) for (a,b) = zip(eachcol(cryst.latvecs), eachcol(recip_vecs))]
    n_max = round.(Int, max_dist ./ box_lengths, RoundUp)

    bonds = Bond[]

    # loop over neighboring cells
    for n1 in -n_max[1]:n_max[1]
        for n2 in -n_max[2]:n_max[2]
            for n3 in -n_max[3]:n_max[3]
                n = SVector(n1, n2, n3)
                
                # loop over all atoms within neighboring cell
                for j in eachindex(cryst.positions)
                    b = Bond(i, j, n)
                    if min_dist <= global_distance(cryst, b) <= max_dist
                        push!(bonds, b)
                    end
                end
            end
        end
    end

    return bonds
end


# Calculate score for a bond. Lower would be preferred.
function score_bond(cryst::Crystal, b::Bond)
    # Favor bonds with fewer nonzero elements in basis matrices J
    Js = basis_for_symmetry_allowed_couplings(cryst, b)
    nnz = [count(abs.(J) .> 1e-12) for J in Js]
    score = Float64(sum(nnz))

    # Favor bonds with smaller unit cell displacements. Positive
    # displacements are slightly favored over negative displacements.
    # Displacements in x are slightly favored over y, etc.
    score += norm((b.n .- 0.1) .* [0.07, 0.08, 0.09])

    # Favor smaller indices and indices where i < j
    score += 1e-2 * (b.i + b.j) + 1e-2 * (b.i < b.j ? -1 : +1)

    return score
end

# Indices of the unique elements in `a`, ordered by their first appearance.
function unique_indices(a)
    map(x->x[1], unique(x->x[2], enumerate(a)))
end

"""    reference_bonds(cryst::Crystal, max_dist)

Returns a full list of bonds, one for each symmetry equivalence class, up to
distance `max_dist`. The reference bond `b` for each equivalence class is
selected according to a scoring system that prioritizes simplification of the
elements in `basis_for_symmetry_allowed_couplings(cryst, b)`."""
function reference_bonds(cryst::Crystal, max_dist::Float64; min_dist=0.0)
    # Bonds, one for each equivalence class
    ref_bonds = Bond[]
    for i in unique_indices(cryst.classes)
        for b in all_bonds_for_atom(cryst, i, max_dist; min_dist)
            if !any(is_related_by_symmetry(cryst, b, b′) for b′ in ref_bonds)
                push!(ref_bonds, b)
            end
        end
    end

    # Sort by distance
    sort!(ref_bonds, by=b->global_distance(cryst, b))

    # Replace each canonical bond by the "best" equivalent bond
    return map(ref_bonds) do rb
        # Find full set of symmetry equivalent bonds
        equiv_bonds = unique([transform(cryst, s, rb) for s in cryst.symops])
        # Take the bond with lowest score
        return argmin(b -> score_bond(cryst, b), equiv_bonds)
    end
end
reference_bonds(cryst::Crystal, max_dist) = reference_bonds(cryst, convert(Float64, max_dist))

"""
    all_symmetry_related_bonds_for_atom(cryst::Crystal, i::Int, b::Bond)

Returns a list of all bonds that start at atom `i`, and that are symmetry
equivalent to bond `b` or its reverse.
"""
function all_symmetry_related_bonds_for_atom(cryst::Crystal, i::Int, b_ref::Bond)
    # Some people will try to model inhomogeneous systems with very large unit
    # cells under the 'P1' spacegroup, such that cryst.symops contains only the
    # identity. In this case, there are only two 'symmetry related couplings':
    # `b_ref` itself and its reverse. Return early for performance.
    if cryst.symops == [SymOp(Mat3(I), zero(Vec3))]
        i == b_ref.i && return [b_ref]
        i == b_ref.j && return [reverse(b_ref)]
        return Bond[]
    end

    bs = Bond[]
    dist = global_distance(cryst, b_ref)
    for b in all_bonds_for_atom(cryst, i, dist; min_dist=dist)
        if is_related_by_symmetry(cryst, b_ref, b)
            push!(bs, b)
        end
    end
    return bs
end

"""
    all_symmetry_related_bonds(cryst::Crystal, b::Bond)

Returns a list of all bonds that are symmetry-equivalent to bond `b` or its
reverse.
"""
function all_symmetry_related_bonds(cryst::Crystal, b_ref::Bond)
    bs = Bond[]
    for i in eachindex(cryst.positions)
        append!(bs, all_symmetry_related_bonds_for_atom(cryst, i, b_ref))
    end
    bs
end

"""    coordination_number(cryst::Crystal, i::Int, b::Bond)

Returns the number times that atom `i` participates in a bond equivalent to `b`.
In other words, the count of bonds that begin at atom `i` and that are
symmetry-equivalent to `b` or its reverse.

Defined as `length(all_symmetry_related_bonds_for_atom(cryst, i, b))`.
"""
function coordination_number(cryst::Crystal, i::Int, b::Bond)
    return length(all_symmetry_related_bonds_for_atom(cryst, i, b))
end
