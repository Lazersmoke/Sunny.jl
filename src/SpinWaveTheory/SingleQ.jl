function cartesian(n) # create two vectors perpendicular to n.
    z = [0,0,1]
    y = [0,1,0]
    c = cross(n,z)
    if norm(c) < 1e-8
        vy = cross(n,y)
    else
        vy = c
    end
    vy = normalize(vy)
    vz = normalize(cross(n,vy))
    return([vy,vz])
end


# planar

function gm_planar!(sys::System, n, x)
    k = x[end-2:end]
    u,v = cartesian(n);
    nspin = length(x) - 3
    phi = x[1:nspin]
    for i in 1:nspin
        if length(sys.Ns) != nspin
            error("Error:The magnetic atoms are not correct")
        end
        sys.dipoles[i] = (u * cos(phi[i]) + v * sin(phi[i])) * sys.κs[i];
    end
    E = spiral_energy(sys,k,n)
    return E
end

function gm_spherical3d!(sys::System, n, x)
    nspin = Int64((length(x)-3)/2)
    k = x[end-2:end]
    MTheta  = x[(1:nspin) .* 2 .- 1]
    MPhi    = x[(1:nspin) .* 2]
	R = Sunny.rotation_between_vectors(n, [0, 0, 1])
    for i in 1:nspin
        if length(MTheta)!=length(sys.Ns)
            error("gm_spherical3d:NumberOfMoments','The number of fitting parameters doesn''t produce the right number of moments!")
        end
        sys.dipoles[i] = R * [sin(MTheta[i])*cos(MPhi[i]); sin(MTheta[i])*sin(MPhi[i]); cos(MTheta[i])] * sys.κs[i]
    end
    E = spiral_energy(sys,k,n)
    return E
end

# The idealized exchange energy for a spiral order of momentum k, ignoring local
# normalization constraints.
function luttinger_tisza_exchange(sys::System, k; ϵ=0)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    Na = natoms(sys.crystal)
    J_k = zeros(ComplexF64, 3, Na, 3, Na)

    for i in 1:natoms(sys.crystal)
        for coupling in sys.interactions_union[i].pair
            (; isculled, bond, bilin) = coupling
            isculled && break

            (; j, n) = bond
            J = exp(2π * im * dot(k, n)) * Mat3(bilin*I)
            J_k[:, i, :, j] += J / 2
            J_k[:, j, :, i] += J' / 2
        end
    end

    J_k = reshape(J_k, 3*Na, 3*Na)
    @assert diffnorm2(J_k, J_k') < 1e-16
    J_k = hermitianpart(J_k)

    E = if iszero(ϵ)
        eigmin(J_k)
    else
        # Estimate the minimum eigenvalue E as a weighted average of all small
        # eigenvalues, E = tr [exp(-β J) J] / tr exp(-β J), where β = 1/ϵ.
        # Finite but small ϵ will regularize the potential energy surface in the
        # viscinity of degenerate eigenvalues. Imposing this smoothness may aid
        # optimization of k with gradient-based methods.
        λs = eigvals(J_k)
        λmin = minimum(λs)
        # Scale all weights exp(-λ/ϵ) by the factor exp(λmin/ϵ). Scaling of
        # weights has no mathematical effect, but avoids numerical overflow.
        ws = @. exp(-(λs-λmin)/ϵ)
        sum(ws .* λs) / sum(ws)
    end

    # Scale minimum eigenvalue E by ∑ᵢSᵢ², which is valid under the L.T.
    # assumption that each spin is locally normalized.
    return E * norm2(sys.κs)
end

# Starting from an initial guess, return the wavevector k that locally minimizes
# `luttinger_tisza_exchange`.
function optimize_luttinger_tisza_exchange(sys::System, k)
    iterations = 1000
    options = Optim.Options(; iterations)
    res = Optim.optimize(k, Optim.ConjugateGradient(), options) do k
        luttinger_tisza_exchange(sys, k; ϵ=1e-8)
    end

    if Optim.converged(res)
        k = Optim.minimizer(res)
        # Wrap components to [0, 1)
        k = wrap_to_unit_cell(Vec3(k); symprec=1e-10)
        return k
    else
        error("Momentum optimization failed to converge within $iterations iterations.")
    end
end

# Starting from an L×L×L Fourier-space grid, find a wavevector `k` that locally
# minimizes `luttinger_tisza_exchange`.
function optimize_luttinger_tisza_exchange(sys::System; L=20)
    ks = [[kx, ky, kz]/L for kx in 0:L-1, ky in 0:L-1, kz in 0:L÷2][:]
    Es = [luttinger_tisza_exchange(sys, k) for k in ks]
    P = sortperm(Es)
    # Take 10 k values with lowest energy
    ks = ks[first(P, 10)]
    println(ks)
    # Locally optimize them
    ks = [optimize_luttinger_tisza_exchange(sys, k) for k in ks]
    # Return the k value with smallest energy
    return findmin(k -> luttinger_tisza_exchange(sys, k), ks)[1]
end


function spiral_energy(sys::System, k, axis)
    @assert sys.mode in (:dipole, :dipole_large_S) "SU(N) mode not supported"
    @assert sys.latsize == (1, 1, 1) "System must have only a single cell"

    E = 0
    for i in 1:natoms(sys.crystal)
        (; onsite, pair) = sys.interactions_union[i]
        Si = sys.dipoles[i]

        # Pair coupling
        for coupling in pair
            (; isculled, bond, bilin, biquad) = coupling
            isculled && break

            (; j, n) = bond
            θ = 2π * dot(k, n)
            R = axis_angle_to_matrix(axis, θ)

            J = bilin
            Sj = sys.dipoles[j]
            E += Si' * (J * R) * Sj
            # Note invariance under global rotation R
            @assert J * R ≈ R * J

            @assert iszero(biquad) "Biquadratic interactions not supported"
        end

        # Onsite coupling
        E += energy_and_gradient_for_classical_anisotropy(Si, onsite)[1]

        # Zeeman coupling
        E -= sys.extfield[i]' * magnetic_moment(sys, i)
    end

    return E
end

function optimagstr(f::Function, xmin, xmax, x0; maxiters=1000) # optimizing function to get minimum energy and propagation factor.
    results = Optim.optimize(f, xmin, xmax, x0, Optim.Fminbox(Optim.BFGS()), Optim.Options(iterations=maxiters))
	println("Ground state Energy(meV) = ", results.minimum)
	opt = Optim.minimizer(results)
    k = opt[end-2:end]
    return k
end

function construct_uniaxial_anisotropy(; n, c20=0., c40=0., c60=0., S)
    # Anisotropy operator in frame of `n`
    O = Sunny.stevens_matrices(S)
    op = c20*O[2, 0] + c40*O[4, 0] + c60*O[6, 0]
    # Anisotropy operator in global frame (rotates n to [0, 0, 1])
    R = Sunny.rotation_between_vectors(n, [0, 0, 1])
    return rotate_operator(op, R)
end




## Dispersion and intensities


function swt_hamiltonian_dipole_singleQ!(H::Matrix{ComplexF64}, swt::SpinWaveTheory, q_reshaped, n::Vector{Float64}, k::Vector{Float64})
    (; sys, data) = swt
    (; local_rotations, stevens_coefs) = data
    N = swt.sys.Ns[1]
    S = (N-1)/2
    L = Sunny.nbands(swt) 
    @assert size(H) == (2L, 2L)
    H .= 0.0 
    
    #Add pairwise bilinear term
    for matom = 1:L
        ints = sys.interactions_union[matom]
        
        for c in ints.pair
            (; bond) = c
            d = bond.n
            θ = (2*π * dot(k,d))
            R = [cos(θ)+(n[1]^2)*(1-cos(θ)) n[1]*n[2]*(1-cos(θ))-n[3]*sin(θ) n[1]*n[3]*(1-cos(θ))+n[2]*sin(θ);
                 n[2]*n[1]*(1-cos(θ))+n[3]*sin(θ) cos(θ)+(n[2]^2)*(1-cos(θ)) n[2]*n[3]*(1-cos(θ))-n[1]*sin(θ);
                 n[3]*n[1]*(1-cos(θ))-n[2]*sin(θ) n[2]*n[3]*(1-cos(θ))+n[1]*sin(θ) cos(θ)+(n[3]^2)*(1-cos(θ))]
            
            
            sub_i, sub_j = bond.i, bond.j
            local_rotations_i = local_rotations[sub_i]
            local_rotations_j = local_rotations[sub_j]
            J = c.bilin
            J = (local_rotations_i * c.bilin * local_rotations_j') ./S
            
            Jij = (J * R + R * J) ./ 2
            phase = exp(2π*im * dot(q_reshaped, d))
            
           
            si = (sys.Ns[sub_i]-1)/2
            sj = (sys.Ns[sub_j]-1)/2  

            

            ui = local_rotations_i[:,1]+im*local_rotations_i[:,2]
            vi = local_rotations_i[:,3]

            uj = local_rotations_j[:,1]+im*local_rotations_j[:,2]
        
            vj = local_rotations_j[:,3]
            
            H[sub_i,sub_j] += (sqrt(si*sj)/2) * (transpose(ui)) * Jij * conj(uj) * phase
            H[sub_i+L,sub_j+L] +=  (sqrt(si*sj)/2) * conj((transpose(ui)) * Jij * conj(uj)) * phase
          
            H[sub_i,sub_j+L] += (sqrt(si*sj)/2) * (transpose(ui) * Jij * uj) * phase
          
            H[sub_j+L,sub_i] +=  (sqrt(si*sj)/2) * conj(transpose(ui) * Jij * uj * phase)
          
            H[sub_i,sub_i] -= sj * transpose(vi) * Jij * vj 
            H[sub_i+L,sub_i+L] -= sj * transpose(vi) * Jij * vj
        end
    end

    H[:,:] = H / 2
    # Add Zeeman term
    (; extfield, gs, units) = sys
    for i in 1:L
        B = units.μB * (Transpose(extfield[1, 1, 1, i]) * gs[1, 1, 1, i]) 
        B′ = (B * local_rotations[i][:, 3]) / 2 
        H[i, i]     += B′
        H[i+L, i+L] += conj(B′)
    end
    
    for i in 1:L
        (; c2, c4, c6) = stevens_coefs[i]
        H[i, i]     += -3S*c2[3] - 40*S^3*c4[5] - 168*S^5*c6[7]
        H[i+L, i+L] += -3S*c2[3] - 40*S^3*c4[5] - 168*S^5*c6[7]
        H[i, i+L]   += +im*(S*c2[5] + 6S^3*c4[7] + 16S^5*c6[9]) + (S*c2[1] + 6S^3*c4[3] + 16S^5*c6[5])
        H[i+L, i]   += -im*(S*c2[5] + 6S^3*c4[7] + 16S^5*c6[9]) + (S*c2[1] + 6S^3*c4[3] + 16S^5*c6[5])
    end
        
    if norm(H-H') > 1e-12
        println("norm(H-H')= ", norm(H-H'))
        throw("H is not hermitian!")
    else
    end 

    Sunny.hermitianpart!(H)
    
    for i = 1:2L
        H[i, i] += 1e-9
    end
end

function dispersion_singleQ(swt::SpinWaveTheory,n::Vector{Float64},k::Vector{Float64}, qs)
    (; sys) = swt
    
    Nm, Ns = length(sys.dipoles), sys.Ns[1] # number of magnetic atoms and dimension of Hilbert space
    Nf = sys.mode == :SUN ? Ns-1 : 1
    nmodes  = Nf * Nm

   
    disp = zeros(Float64, nmodes, length(qs),3)

    for (iq, q) in enumerate(qs)
        for branch = 1:3    # 3 branch corresponds to K,K+Q and K-Q modes of incommensurate spin structures.
            H = zeros(ComplexF64, 2nmodes, 2nmodes)
            V = zeros(ComplexF64, 2nmodes, 2nmodes)
            q_reshaped = Sunny.to_reshaped_rlu(swt.sys, q)
            if sys.mode == :SUN
                error("SingleQ calculation for SUN is not yet implemented")
            else
                @assert sys.mode in (:dipole, :dipole_large_S)
                swt_hamiltonian_dipole_singleQ!(H, swt, q_reshaped .+ (branch - 2) .* k, n, k)
            end
            try
                    view(disp, :, iq,branch) .= Sunny.bogoliubov!(V, H)
            catch e
                    error("Instability at wavevector q = $q")
            end
        end
    end

    return  disp 
end


struct DipoleSingleQSpinWaveIntensityFormula{T}
    n :: Vector{Float64}
    k :: Vector{Float64}
    string_formula :: String
    kernel :: Union{Nothing,Function}
    calc_intensity :: Function
end

function Base.show(io::IO, ::DipoleSingleQSpinWaveIntensityFormula{T}) where T
    print(io,"SpinWaveIntensityFormula{$T}")
end

function Base.show(io::IO, ::MIME"text/plain", formula::DipoleSingleQSpinWaveIntensityFormula{T}) where T
    printstyled(io, "Quantum Scattering Intensity Formula\n"; bold=true, color=:underline)

    formula_lines = split(formula.string_formula, '\n')

    if isnothing(formula.kernel)
        println(io, "At any Q and for each band ωᵢ = εᵢ(Q), with S = S(Q,ωᵢ):\n")
        intensity_equals = "  Intensity(Q,ω) = ∑ᵢ δ(ω-ωᵢ) "
    else
        println(io, "At any (Q,ω), with S = S(Q,ωᵢ):\n")
        intensity_equals = "  Intensity(Q,ω) = ∑ᵢ Kernel(ω-ωᵢ) "
    end
    separator = '\n' * repeat(' ', textwidth(intensity_equals))
    println(io, intensity_equals, join(formula_lines, separator))
    println(io)
    if isnothing(formula.kernel)
        println(io,"BandStructure information (ωᵢ and intensity) reported for each band")
    else
        println(io,"Intensity(ω) reported")
    end
end

delta_function_kernel = nothing



"""
    intensity_formula([swt or sc], contraction_mode::Symbol)

Sunny has several built-in formulas that can be selected by setting `contraction_mode` to one of these values:

- `:trace` (default), which yields ``\\operatorname{tr} 𝒮(q,ω) = ∑_α 𝒮^{αα}(q,ω)``
- `:perp`, which contracts ``𝒮^{αβ}(q,ω)`` with the dipole factor ``δ_{αβ} - q_{α}q_{β}``, returning the unpolarized intensity.
- `:full`, which will return all elements ``𝒮^{αβ}(𝐪,ω)`` without contraction.
"""
function intensity_formula_SingleQ(swt::SpinWaveTheory, k::Vector{Float64}, n::Vector{Float64}, mode::Symbol; kwargs...)
    contractor, string_formula = Sunny.contractor_from_mode(swt, mode)
    intensity_formula_SingleQ(swt, k, n,  contractor; string_formula, kwargs...)
end

function intensity_formula_SingleQ(swt::SpinWaveTheory, k::Vector{Float64}, n::Vector{Float64}, contractor::Sunny.Contraction{T}; kwargs...) where T
    intensity_formula_SingleQ(swt, k, n, Sunny.required_correlations(contractor); return_type = T,kwargs...) do ks,ωs,correlations
        intensity = Sunny.contract(correlations, ks, contractor)
    end
end


"""
    formula = intensity_formula(swt::SpinWaveTheory; kernel = ...)

Establish a formula for computing the scattering intensity by diagonalizing
the hamiltonian ``H(q)`` using Linear Spin Wave Theory.

If `kernel = delta_function_kernel`, then the resulting formula can be used with
[`intensities_bands`](@ref).

If `kernel` is an energy broadening kernel function, then the resulting formula can be used with [`intensities_broadened`](@ref).
Energy broadening kernel functions can either be a function of `Δω` only, e.g.:

    kernel = Δω -> ...

or a function of both the energy transfer `ω` and of `Δω`, e.g.:

    kernel = (ω,Δω) -> ...

The integral of a properly normalized kernel function over all `Δω` is one.
"""

function intensity_formula_SingleQ(f::Function,swt::SpinWaveTheory,k::Vector{Float64},n::Vector{Float64},corr_ix::AbstractVector{Int64}; kernel::Union{Nothing,Function},
                           return_type=Float64, string_formula="f(Q,ω,S{α,β}[ix_q,ix_ω])", 
                           formfactors=nothing)
    (; sys, data, observables) = swt
    Nm, Ns = length(sys.dipoles), sys.Ns[1] # number of magnetic atoms and dimension of Hilbert space
    nmodes = Sunny.nbands(swt) # k, k+Q, k-Q


    H = zeros(ComplexF64, 2*nmodes, 2*nmodes)
    T = zeros(ComplexF64, 2*nmodes, 2*nmodes,3)
    tmp = zeros(ComplexF64, 2*nmodes, 2*nmodes)

    disp = zeros(Float64, nmodes, 3)
    intensity = zeros(return_type, nmodes,3)
    S = zeros(ComplexF64,3,3,nmodes,3)

    FF = zeros(ComplexF64, Nm)
    #intensity = zeros(return_type, nmodes,3)

    # Expand formfactors for symmetry classes to formfactors for all atoms in
    # crystal
    ff_atoms = Sunny.propagate_form_factors_to_atoms(formfactors, swt.sys.crystal)
    
    # Upgrade to 2-argument kernel if needed
    kernel_edep = if isnothing(kernel)
        nothing
    else
        try
            kernel(0.,0.)
            kernel
        catch MethodError
            (ω,Δω) -> kernel(Δω)
        end
    end



    # In Spin Wave Theory, the Hamiltonian depends on momentum transfer `q`.
    # At each `q`, the Hamiltonian is diagonalized one time, and then the
    # energy eigenvalues can be reused multiple times. To facilitate this,
    # `I_of_ω = calc_intensity(swt,q)` performs the diagonalization, and returns
    # the result either as:
    #
    #   Delta function kernel --> I_of_ω = (eigenvalue,intensity) pairs
    #
    #   OR
    #
    #   Smooth kernel --> I_of_ω = Intensity as a function of ω
    #
    calc_intensity = function(swt::SpinWaveTheory, q::Sunny.Vec3)
        # This function, calc_intensity, is an internal function to be stored
        # inside a formula. The unit system for `q` that is passed to
        # formula.calc_intensity is an implementation detail that may vary
        # according to the "type" of a formula. In the present context, namely
        # LSWT formulas, `q` is given in RLU for the original crystal. This
        # convention must be consistent with the usage in various
        # `intensities_*` functions defined in LinearSpinWaveIntensities.jl.
        # Separately, the functions calc_intensity for formulas associated with
        # SampledCorrelations will receive `q_absolute` in absolute units.
        nx = [0 -n[3] n[2];n[3] 0 -n[1];-n[2] n[1] 0]
        R1 = (1/2) .* (I - im .* nx - n * Transpose(n))
        R2 = n*Transpose(n)
        
        q_reshaped = Sunny.to_reshaped_rlu(swt.sys, q)
        q_absolute = swt.sys.crystal.recipvecs * q_reshaped 

        for branch = 1:3   # 3 branch corresponds to K,K+Q and K-Q modes of incommensurate spin structures.
            if sys.mode == :SUN
                error("SingleQ calculation for SUN is not yet implemented")
            else
                @assert sys.mode in (:dipole, :dipole_large_S)
                
                swt_hamiltonian_dipole_singleQ!(H, swt, q_reshaped + (branch-2)*k, n, k)
            
                disp[:,branch] = try
                    Sunny.bogoliubov!(tmp, H)
                catch e
                    error("Instability at wavevector q = $q")
                end

                T[:,:,branch] = tmp
            end
        end

        for i = 1:Nm
            @assert Nm == Sunny.natoms(sys.crystal)
            # TODO: move form factor into `f`, then delete this rescaling
            if formfactors == nothing
                FF[i] = 1.0
            else
                FF[i] = Sunny.compute_form_factor(ff_atoms[i], q_absolute⋅q_absolute)
            end
        end
       
        R = data.local_rotations
        Y = zeros(ComplexF64,nmodes,nmodes,3,3)
        Z = zeros(ComplexF64,nmodes,nmodes,3,3)
        V = zeros(ComplexF64,nmodes,nmodes,3,3)
        W = zeros(ComplexF64,nmodes,nmodes,3,3)
            for α in 1:3, β in 1:3
                for i in 1:nmodes, j in 1:nmodes
                    si = (sys.Ns[i]-1)/2
                    sj = (sys.Ns[j]-1)/2
                    R_i = R[i]
                    R_j = R[j]
                    ui = R_i[:,1]+im*R_i[:,2]
                    uj = R_j[:,1]+im*R_j[:,2]
                    ti = sys.crystal.positions[i]
                    tj = sys.crystal.positions[j]
                    phase = exp(2π * im*dot(q_reshaped,(ti-tj)))
                    Y[i,j,α,β] = FF[i]*FF[j]*sqrt(si*sj) * (ui[α] * conj(uj[β])) * (phase)
                    Z[i,j,α,β] = FF[i]*FF[j]*sqrt(si*sj) * (ui[α] * uj[β]) * (phase)
                    V[i,j,α,β] = FF[i]*FF[j]*sqrt(si*sj) * (conj(ui[α]) * conj(uj[β])) * (phase)
                    W[i,j,α,β] = FF[i]*FF[j]*sqrt(si*sj) * (conj(ui[α]) * uj[β]) * (phase)
                end
            end 
        YZVW = [[Y Z];[V W]]

        for branch = 1:3
            for band = 1:nmodes
                
                corrs = if sys.mode == :SUN
                    error("SingleQ calculation for SUN is not yet implemented")
                else
                    @assert sys.mode in (:dipole, :dipole_large_S)
                    for α in 1:3
                        for β in 1:3
                            A = T[:,:,branch]' * YZVW[:,:,α,β] * T[:,:,branch]
                        
                            S[α,β,band,branch] = (1/(2*Nm)) * A[band,band] 
                            
                        end
                    end
                end
            end
        end
        
        avg = (S -> 1/2 * (S .- nx * S * nx .+ (R2 - I) * S * R2 .+ R2 * S * (R2 -I) .+ R2 * S * R2))
        
        for band = 1:nmodes
            S[:,:,band,1] = avg(S[:,:,band,1]) * conj(R1)
            S[:,:,band,2] = avg(S[:,:,band,2]) * R2
            S[:,:,band,3] = avg(S[:,:,band,3]) * R1
        end
        
        for branch = 1:3
            for band = 1:nmodes
                
                @assert observables.observable_ixs[:Sx] == 1
                @assert observables.observable_ixs[:Sy] == 2
                @assert observables.observable_ixs[:Sz] == 3

                corrs = Vector{ComplexF64}(undef, Sunny.num_correlations(observables))
                for (ci,i) in observables.correlations
                    (α,β) = ci.I

                    corrs[i] = S[α,β,band,branch]
                end
                
                intensity[band,branch] = f(q_absolute, disp[band,branch], corrs[corr_ix])
                
            end
        end

    
        # Return the result of the diagonalization in an appropriate
        # format based on the kernel provided
        if isnothing(kernel)
            # Delta function kernel --> (eigenvalue,intensity) pairs

            # If there is no specified kernel, we are done: just return the
            # BandStructure
            return Sunny.BandStructure{3*nmodes,return_type}(disp,intensity)

        else
            # Smooth kernel --> Intensity as a function of ω (or a list of ωs)
            return function(ω)
                is = Array{return_type}(undef,length(ω),nmodes,3)
                for branch = 1:3
                    for band = 1:nmodes
                        is[:,band,branch] = intensity[band,branch]' .* kernel_edep.(disp[band,branch]', ω .- disp[band,branch]')
                    end 
                end
                is
            end
        end
    end
    output_type = isnothing(kernel) ? Sunny.BandStructure{nmodes,return_type} : return_type
    DipoleSingleQSpinWaveIntensityFormula{output_type}(n,k,string_formula,kernel_edep,calc_intensity)
end


function intensities_bands_SingleQ(swt::SpinWaveTheory, ks, formula::DipoleSingleQSpinWaveIntensityFormula)
    if !isnothing(formula.kernel)
        # This is only triggered if the user has explicitly specified a formula with e.g. kT
        # corrections applied, but has not disabled the broadening kernel.
        error("intensities_bands: Can't compute band intensities if a broadening kernel is applied.\nTry intensity_formula(...; kernel = delta_function_kernel)")
    end

    ks = Sunny.Vec3.(ks)
    nmodes = Sunny.nbands(swt)

    # Get the type parameter from the BandStructure
    return_type = typeof(formula).parameters[1].parameters[2]

    band_dispersions = zeros(Float64,length(ks),3*nmodes)
    band_intensities = zeros(return_type,length(ks),3*nmodes)
    #for branch = 1:3
        for kidx in CartesianIndices(ks)
            band_structure = formula.calc_intensity(swt, ks[kidx])

            # Place the BandStructure at each point into its location in the array
            band_dispersions[kidx,:] .= band_structure.dispersion
            band_intensities[kidx,:] .= band_structure.intensity
        end
    #end
    return band_dispersions, band_intensities
end


function intensities_broadened_SingleQ(swt::SpinWaveTheory, ks, ωvals, formula)
    ks = Sunny.Vec3.(ks)
    num_ω = length(ωvals)
    nmodes = Sunny.nbands(swt)
    return_type = typeof(formula).parameters[1]
    if return_type <: Sunny.BandStructure 
        # This only happens if the user sets `kernel = delta_function_kernel`
        error("intensities_broadened: Can't compute broadened intensities without a finite-width kernel.\nTry: intensity_formula(...; kernel = lorentzian(0.05))")
    end

    is = zeros(size(ks)..., num_ω,nmodes,3)

    # Compute the intensity at each (k,ω) pair
    for kidx in CartesianIndices(ks)
                intensity_as_function_of_ω = formula.calc_intensity(swt,ks[kidx])
                is[kidx,:,:,:] .= intensity_as_function_of_ω(ωvals)
    end 
    return is
end