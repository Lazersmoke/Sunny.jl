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
        if i <= j
            Q[i,j] = real(M[i,j] + M[j,i])
        else
            Q[i,j] = imag(M[j,i] - M[i,j])
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
        if i <= j
            out[i] += Z[j] * dE_dT[k]
            out[j] += Z[i] * dE_dT[k]
        else
            out[i] += im * Z[j] * dE_dT[k]
            out[j] -= im * Z[i] * dE_dT[k]
        end
    end
    out
end

function dipolar_part(Q)
    error("NYI")
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

function spin_multipoles_representation(; N)
    S = spin_matrices(; N)
    T = multipolar_matrices(; N)
    M = zeros(ComplexF64,N^2-1,N)
    for i = 1:(N^2-1), j = 1:3
        M[i,j] = tr(T[i] * S[j])
    end
    function ρ(R)
        M * R * M'
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

