"""
    spin_matrices(; N)

Constructs the three spin operators, i.e. the generators of SU(2), in the
`N`-dimensional irrep. See also [`spin_operators`](@ref), which determines the
appropriate value of `N` for a given site index.
"""
function spin_matrices(; N::Int)
    if N == 0
        return fill(zeros(ComplexF64,0,0), 3)
    end

    S = (N-1)/2
    j = 1:N-1
    off = @. sqrt(2(S+1)*j - j*(j+1)) / 2

    Sx = diagm(1 => off, -1 => off)
    Sy = diagm(1 => -im*off, -1 => +im*off)
    Sz = diagm(S .- (0:N-1))
    return [Sx, Sy, Sz]
end

# Returns ⟨Z|Sᵅ|Z⟩
@generated function expected_spin(Z::CVec{N}) where N
    S = spin_matrices(; N)
    elems_x = SVector{N-1}(diag(S[1], 1))
    elems_z = SVector{N}(diag(S[3], 0))
    lo_ind = SVector{N-1}(1:N-1)
    hi_ind = SVector{N-1}(2:N)

    return quote
        $(Expr(:meta, :inline))
        c = Z[$lo_ind]' * ($elems_x .* Z[$hi_ind])
        nx = 2real(c)
        ny = 2imag(c)
        nz = real(Z' * ($elems_z .* Z))
        Vec3(nx, ny, nz)
    end
end

function expected_multipolar_moments(Z::CVec{N}) where N
    M = Z * Z'
    Q = zeros(Float64,N,N)
    for i = 1:N, j = 1:N
        if i == j
            if i == N
                continue
            end
            Q[i,i] = M[i,i]/sqrt(2)
            Q[i+1,i+1] = -M[i+1,i+1]/sqrt(2)
        elseif i < j
            Q[i,j] = real(M[i,j] + M[j,i])/sqrt(2)
        else
            Q[i,j] = imag(M[j,i] - M[i,j])/sqrt(2)
        end
    end
    Q[1:(N^2 - 1)]
end

function multipolar_generators_times_Z(dE_dT,Z::CVec{N}) where N
    out = MVector{N,ComplexF64}(undef)
    out .= 0
    li = LinearIndices(zeros(N,N))
    for i = 1:N, j = 1:N
        k = li[i,j]
        if k == N^2
          break
        end
        if i == j
            out[i] += Z[i] * dE_dT[k]/sqrt(2)
            out[i+1] -= Z[i+1] * dE_dT[k]/sqrt(2)
        elseif i < j
            out[i] += Z[j] * dE_dT[k]/sqrt(2)
            out[j] += Z[i] * dE_dT[k]/sqrt(2)
        else
            out[i] += im * Z[j] * dE_dT[k]/sqrt(2)
            out[j] -= im * Z[i] * dE_dT[k]/sqrt(2)
        end
    end
    out
end

function dipolar_part(Q)
    error("NYI")
end

function physical_basis_su3()
    S = 1
    dipole_operators = spin_matrices(; N=2S+1)
    Sx, Sy, Sz = dipole_operators

    # we choose a particular basis of the quadrupole operators listed in Appendix B of *Phys. Rev. B 104, 104409*
    quadrupole_operators = Vector{Matrix{ComplexF64}}(undef, 5)
    quadrupole_operators[1] = -(Sx * Sz + Sz * Sx)
    quadrupole_operators[2] = -(Sy * Sz + Sz * Sy)
    quadrupole_operators[3] = Sx * Sx - Sy * Sy
    quadrupole_operators[4] = Sx * Sy + Sy * Sx
    quadrupole_operators[5] = √3 * Sz * Sz - 1/√3 * S * (S+1) * I

    [dipole_operators; quadrupole_operators]
end

function multipolar_matrices(; N)
    T = Vector{Matrix{ComplexF64}}(undef,N^2-1)
    li = LinearIndices(zeros(N,N))

    for i = 1:(N^2 - 1)
        T[i] = zeros(ComplexF64,N,N)
        v = zeros(ComplexF64,N^2-1)
        v[i] = 1
        for j = 1:N
            Z = zeros(ComplexF64,N)
            Z[j] = 1
            T[i][:,j] = multipolar_generators_times_Z(v,SVector{N,ComplexF64}(Z))
        end
    end
    T
end

function multipolar_components(S)
    N = size(S,1)
    T = multipolar_matrices(; N)
    M = zeros(ComplexF64,N^2-1)
    for i = 1:(N^2-1)
        M[i] = tr(T[i]' * S) / tr(T[i]' * T[i])
    end
    M
end

function show_inner_products(M)
  l = length(M)
  p = zeros(ComplexF64,l,l)
  for i = 1:l, j = 1:l
    p[i,j] = tr(M[i]' * M[j])
  end
  p
end

function spin_multipoles_representation(; N, T = multipolar_matrices(; N))
    # Make a matrix A = A', and subtract the trace
    traceless_symmetrize(x) = (x+x')./2 - tr(x + x')/(2N) * I(N)
    S = spin_matrices(; N)

    # Build the matrix
    #
    #   S[qs[1]] * S[qs[2]] * ... * S[qs[end]]
    #
    # which is a product of spin matrices
    function make_polynomial_matrix(S,qs)
        prod_matrix = I(N)
        for i in qs
            prod_matrix = prod_matrix * S[i]
        end
        traceless_symmetrize(prod_matrix)
    end

    # This will hold a basis for su(N) [dimension (N^2-1)] where each basis vector
    # is a product of spin matrices, see above.
    spin_polynomial_matrices = Vector{Matrix{ComplexF64}}(undef,N^2-1)
    # Holds the qs for each basis vector:
    spin_polynomials = Vector{Tuple}(undef,N^2-1)

    # Same as spin_polynomial_matrices, but flattened
    flat_matrix = Matrix{ComplexF64}(undef,N^2,N^2-1)
    flat_matrix .= 0

    # We now iteratively construct the aforementioned basis.
    # km keeps track of our progress
    km = 0

    # d = length(qs) is the number of factors in the product
    for d = 1:(N-1)
        # Loop over each factor being (Sx,Sy,Sz)
        for qs = CartesianIndices(ntuple(i -> 3,d))
            if d == 2 && qs.I == (3,3) # Discard Sz^2 explicitly (only applies to N > 2)
                continue
            end
            candidate_basis_vector = make_polynomial_matrix(S,qs.I)

            # If the candidate is linearly independent (increases rank by one)
            # then add it to the basis
            if rank([flat_matrix candidate_basis_vector[:]]) == rank(flat_matrix) + 1
                km = km + 1
                @assert km <= N^2 - 1
                flat_matrix[:,km] = candidate_basis_vector[:]
                spin_polynomial_matrices[km] = candidate_basis_vector
                spin_polynomials[km] = qs.I
            end
        end
    end

    # Assert that we have all (and only all) of the basis vectors
    @assert km == (N^2 - 1)

    #foreach(display,spin_polynomial_matrices)
    #display(spin_polynomials)

    # Write the spin polynomial matrices in terms of the multipolar_matrices basis:
    # spin_polynomial_matrices[j] = M[i,j] * T[i]
    # Like (B1) of **PhysRevB.104.104409**
    M = zeros(ComplexF64,N^2-1,N^2-1)
    for i = 1:(N^2-1), j = 1:(N^2-1)
        M[i,j] = tr(T[i]' * spin_polynomial_matrices[j]) / sqrt(tr(T[i]' * T[i]))
    end

    # The spin matrix products all transform in a known way under physical space
    # rotations, so we perform the rotation on the spin product matrices, and
    # transform that rotation to construct the action on the multipolar_matrices
    rot_spin_polynomial_matrices = Vector{Matrix{ComplexF64}}(undef,N^2-1)
    function ρ(R)
        Srot = Vector{Matrix{ComplexF64}}(undef,3)
        for i = 1:3
            Srot[i] = zeros(ComplexF64,N,N)
            for j = 1:3
                Srot[i] += R[i,j] * S[j]
            end
        end
        for k = 1:km
            rot_spin_polynomial_matrices[k] = make_polynomial_matrix(Srot,spin_polynomials[k])
        end

        # As above:
        # rot_spin_polynomial_matrices[j] = Mrot[i,j] * T[i]
        Mrot = zeros(ComplexF64,N^2-1,N^2-1)
        for i = 1:(N^2-1), j = 1:(N^2-1)
            Mrot[i,j] = tr(T[i]' * rot_spin_polynomial_matrices[j]) / sqrt(tr(T[i]' * T[i]))
        end

        # M,Mrot ~ [multipole operators] x [spin polynomial operators]
        # inv(M),inv(Mrot) ~ [spin polynomial operators] x [multipole operators]
        # J ~ [multipole operators (rotated)] x [multipole operators]
        #
        # and we act as ρ(R) * J * ρ(R')

        #inv(Mrot) * M # ~ [spin polynomial operators (rotated)] x [spin polynomial operators]
        real(Mrot * inv(M)) # ~ [multipole operators (rotated)] x [multipole operators]
    end
end


# Find a ket (up to an irrelevant phase) that corresponds to a pure dipole.
# TODO, we can do this faster by using the exponential map of spin operators,
# expressed as a polynomial expansion,
# http://www.emis.de/journals/SIGMA/2014/084/
ket_from_dipole(_::Vec3, ::Val{0}) :: CVec{0} = zero(CVec{0})
function ket_from_dipole(dip::Vec3, ::Val{N}) :: CVec{N} where N
    S = spin_matrices(; N)
    λs, vs = eigen(dip' * S)
    return CVec{N}(vs[:, argmax(real.(λs))])
end

# Applies the time-reversal operator to the coherent spin state |Z⟩, which
# effectively negates the expected spin dipole, ⟨Z|Sᵅ|Z⟩ → -⟨Z|Sᵅ|Z⟩.
flip_ket(_::CVec{0}) = CVec{0}()
function flip_ket(Z::CVec{N}) where N
    # Per Sakurai (3rd ed.), eq. 4.176, the time reversal operator has the
    # action T[Z] = exp(-i π Sʸ) conj(Z). In our selected basis, the operator
    # exp(-i π Sʸ) can be implemented by flipping the sign of half the
    # components and then reversing their order.
    parity = SVector{N}(1-2mod(i,2) for i=0:N-1)
    return reverse(parity .* conj(Z))
end

