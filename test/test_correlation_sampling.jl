@testitem "Available Energies Dirac Identity" begin
     # Create a dummy SampledCorrelations object
    latsize = (1,1,1)
    cryst = Sunny.fcc_primitive_crystal()
    sys = System(cryst, latsize, [SpinInfo(1; S = 1/2, g=2)], :SUN; seed = 0)
    dt = 0.1
    sc = dynamical_correlations(sys; Δt = dt, ωmax = 10.0, nω=100)

    ωs = available_energies(sc;negative_energies=true)
    dts = 0:(dt * sc.measperiod):3
    vals = sum(exp.(im .* ωs .* dts'),dims = 1)[:]

    # Verify it made a delta function
    @test vals[1] ≈ length(ωs)
    @test all(isapprox.(0,vals[2:end];atol = 1e-12))
end

@testitem "Asymmetric Correlations" begin
    using FFTW

    # Create a dummy SampledCorrelations object
    latsize = (1,1,1)
    cryst = Sunny.fcc_primitive_crystal()
    sys = System(cryst, latsize, [SpinInfo(1; S = 1/2, g=2)], :SUN; seed = 0)
    sc = dynamical_correlations(sys; Δt = 0.1, ωmax = 10.0, nω=100, observables = [:A => [1. 0im; 0 1], :B => [1. 0im; 0 1]])

    # Fill the sc.samplebuf with test signals.
    # The test signals are asymmetrically correlated.
    # This code replaces Sunny.new_sample!
    time_T = size(sc.samplebuf,6)
    time_2T = 2time_T - 1
    ts = range(0,1,length = time_2T+1)[1:end-1]
    As = exp.(-(ts .- 0.15).^2 ./ (2 * 0.05^2))[1:time_T]
    Bs = exp.(-(ts .- 0.25).^2 ./ (2 * 0.02^2))[1:time_T]
    sc.samplebuf[1,1,1,1,1,:] .= As
    sc.samplebuf[2,1,1,1,1,:] .= Bs

    # Run the Sunny-internal function which computes the correlations
    Sunny.accum_sample!(sc)

    # Retrieve the correlations in direct time.
    # The sc.data is normalized such that FFTW.ifft (which includes
    # a factor of 1/size(sc.data,7)) will give the correct time-domain correlations.
    # TODO: The correlations tested here are real--make them complex
    real_data = real(FFTW.ifft(sc.data,7))

    # Reference calculation
    q11 = zeros(time_2T)
    q12 = zeros(time_2T)
    q21 = zeros(time_2T)
    q22 = zeros(time_2T)
    offsets = Int64.(fftfreq(time_2T,time_2T))
    for i = eachindex(offsets)
        # Uses all available data
        offset = offsets[i]
        valid_starting_times = max(1,1-offset):min(time_T,time_T - offset)
        n = length(valid_starting_times)
        for tau = valid_starting_times
            q11[i] += As[offset+tau] * As[tau] / n
            q12[i] += As[offset+tau] * Bs[tau] / n
            q21[i] += Bs[offset+tau] * As[tau] / n
            q22[i] += Bs[offset+tau] * Bs[tau] / n
        end
    end

    # Compare to both positive and negative time correlations
    @test isapprox(q11,real_data[sc.observables.correlations[CartesianIndex(1,1)],1,1,1,1,1,:];atol = 1e-8)
    @test isapprox(q12,real_data[sc.observables.correlations[CartesianIndex(1,2)],1,1,1,1,1,:];atol = 1e-8)
    @test isapprox(q21,real_data[sc.observables.correlations[CartesianIndex(2,1)],1,1,1,1,1,:];atol = 1e-8)
    @test isapprox(q22,real_data[sc.observables.correlations[CartesianIndex(2,2)],1,1,1,1,1,:];atol = 1e-8)
end

@testitem "Correlation sampling" begin
    using LinearAlgebra

    function simple_model_fcc(; mode, seed=111)
        latsize = (4, 4, 4)
        J = 1.0

        # FCC with nonstandard, primitive lattice vectors
        latvecs = [[1, 1, 0] [0, 1, 1] [1, 0, 1]] / 2
        positions = [[0, 0, 0]]
        cryst = Crystal(latvecs, positions)

        S = mode==:SUN ? 1/2 : 1
        κ = mode==:SUN ? 2 : 1
        sys = System(cryst, latsize, [SpinInfo(1; S, g=2)], mode; seed)
        sys.κs .= κ
        set_exchange!(sys, J, Bond(1, 1, [1, 0, 0]))
        return sys
    end

    function thermalize_simple_model!(sys; kT)
        Δt = 0.05  # Time step for thermalization
        λ  = 0.1
        nsteps = 1000  # Number of steps between MC samples
        langevin = Langevin(Δt; kT, λ)
        for _ in 1:nsteps
            step!(sys, langevin)
        end
    end
    

    # Test sum rule with custom observables 
    sys = simple_model_fcc(; mode=:SUN)
    thermalize_simple_model!(sys; kT=0.1)
    S = spin_matrices(1/2)
    observables = Dict(:Sx => S[1], :Sy => S[2], :Sz => S[3])
    sc = dynamical_correlations(sys; nω=100, ωmax=10.0, Δt=0.1, apply_g=false, observables)
    add_sample!(sc, sys)
    qgrid = available_wave_vectors(sc)
    formula = intensity_formula(sc,:trace)
    Sqw = intensities_interpolated(sc, qgrid, formula; negative_energies=true)
    # To get the time-domain correlations from S(q,w), do FFTW.ifft (which includes a 1/N).
    # As a shortcut to get just the equal-time correlations, we evaluate the ifft manually at t=0.
    equal_time_correlations_from_Sqw(Sqw) = sum(Sqw,dims=4) ./ size(Sqw,4)
    expected_sum_rule = prod(sys.latsize) * norm(sys.dipoles[1])^2 # NS^2 classical sum rule
    @test isapprox(sum(equal_time_correlations_from_Sqw(Sqw)), expected_sum_rule; atol=1e-12)

    # Test sum rule with default observables in dipole mode 
    sys = simple_model_fcc(; mode=:dipole)
    thermalize_simple_model!(sys; kT=0.1)
    sc = dynamical_correlations(sys; Δt=0.1, nω=100, ωmax=10.0, apply_g=false)
    add_sample!(sc, sys)
    trace_formula = intensity_formula(sc,:trace)
    Sqw = intensities_interpolated(sc, qgrid, trace_formula; negative_energies=true)
    total_intensity_trace = sum(Sqw)
    expected_sum_rule = prod(sys.latsize) * norm(sys.dipoles[1])^2 # NS^2 classical sum rule
    @test isapprox(sum(equal_time_correlations_from_Sqw(Sqw)), expected_sum_rule; atol=1e-12)


    # Test perp reduces intensity
    perp_formula = intensity_formula(sc,:perp)
    vals = intensities_interpolated(sc, qgrid, perp_formula; negative_energies=true) 
    total_intensity_unpolarized = sum(vals)
    @test total_intensity_unpolarized < total_intensity_trace


    # Test diagonal elements are approximately real (at one wave vector)
    diag_elems = [(α,α) for α in keys(sc.observables.observable_ixs)]
    formula_imaginary_parts = intensity_formula(sc,diag_elems) do k,ω,corr
        sum(abs.(imag.(corr)))
    end
    intensities_symmetric = intensities_interpolated(sc, [(0.25, 0.5, 0)], formula_imaginary_parts)
    @test sum(imag(intensities_symmetric)) < 1e-15


    # Test form factor correction works and is doing something. ddtodo: add example with sublattice
    formfactors = [FormFactor("Fe2")]
    vals = intensities_interpolated(sc, qgrid, intensity_formula(sc,:trace; formfactors); negative_energies=true)
    total_intensity_ff = sum(vals)
    @test total_intensity_ff != total_intensity_trace


    # Test path function and interpolation working (no correctness implied here)
    qs, _ = reciprocal_space_path(sc.crystal, [(0, 0, 0), (0, 1, 0), (1, 1, 0)], 20)
    vals = intensities_interpolated(sc, qs, perp_formula; interpolation=:linear)
    @test length(size(vals)) == 2 


    # Test static from dynamic intensities working
    static_vals = instant_intensities_interpolated(sc, qgrid, trace_formula; negative_energies=true)
    total_intensity_static = sum(static_vals)
    @test isapprox(total_intensity_static, total_intensity_trace; atol=1e-9)  # Order of summation can lead to very small discrepancies

    # Test instant intensities working
    sys = simple_model_fcc(; mode=:dipole)
    thermalize_simple_model!(sys; kT=0.1)
    ic = instant_correlations(sys; apply_g=false)
    add_sample!(ic, sys)
    true_static_vals = instant_intensities_interpolated(ic, qgrid, intensity_formula(ic,:trace))
    true_static_total = sum(true_static_vals)
    @test isapprox(true_static_total / prod(sys.latsize), 1.0; atol=1e-12)
end

@testitem "Merge correlations" begin
    # Set up a system.
    sys = System(Sunny.diamond_crystal(), (2,2,2), [SpinInfo(1; S=3/2, g=2)], :dipole, seed=101)
    set_exchange!(sys, 0.6498, Bond(1, 3, [0,0,0]))
    randomize_spins!(sys)

    # Set up Langevin sampler.
    Δt_langevin = 0.07 
    langevin = Langevin(Δt_langevin; kT=0.1723, λ=0.1)

    sc0 = dynamical_correlations(sys; nω=25, ωmax=5.5, Δt=2Δt_langevin, calculate_errors=true)
    sc1 = dynamical_correlations(sys; nω=25, ωmax=5.5, Δt=2Δt_langevin, calculate_errors=true)
    sc2 = dynamical_correlations(sys; nω=25, ωmax=5.5, Δt=2Δt_langevin, calculate_errors=true)

    for _ in 1:4_000
        step!(sys, langevin)
    end
    add_sample!(sc0, Sunny.clone_system(sys))
    add_sample!(sc1, Sunny.clone_system(sys))

    for _ in 1:2
        for _ in 1:4_000
            step!(sys, langevin)
        end
        add_sample!(sc0, Sunny.clone_system(sys))
        add_sample!(sc2, Sunny.clone_system(sys))
    end

    # Merge correlations and check if result equal to running calculation.
    sc_merged = merge_correlations([sc1, sc2])
    @test sc0.data ≈ sc_merged.data
    @test sc0.M ≈ sc_merged.M
end

@testitem "Sampled correlations reference" begin
    sys = System(Sunny.diamond_crystal(), (2,3,4), [SpinInfo(1; S=3/2, g=2)], :dipole, seed=101)
    set_exchange!(sys, 0.6498, Bond(1, 3, [0,0,0]))
    randomize_spins!(sys)

    Δt_langevin = 0.07 
    kT = 0.1723
    λ  = 0.1
    langevin = Langevin(Δt_langevin; kT, λ)

    # Thermalize
    for _ in 1:4000
        step!(sys, langevin)
    end

    sc = dynamical_correlations(sys; nω=10, ωmax=5.5, Δt=2Δt_langevin)
    add_sample!(sc, sys)
    qs = [[0.0, 0.0, 0.0], [0.2, -0.4, 0.1]]
    data = intensities_interpolated(sc, qs, intensity_formula(sc,:trace; kT); interpolation=:linear)
    
    refdata = [1.8923137799435257 -1.5731157122871734e-15 -7.618183477999628e-16 2.4258182290582965e-15 2.663555286833582e-15 -2.378171804276773e-15 1.4269030327057308e-15 -1.997664243521173e-15 -4.756343436779901e-16 -1.819301364566135e-15; 0.033223462988952464 0.0565912610212458 0.1616375644454015 4.211237061899472 3.1064676304451533 5.792222570573932 5.536484159910247 0.551596926234539 0.27194613622683184 0.24232982609989023]

    # Compare with reference 
    @test isapprox(data, size(sc.data,7) * refdata / Sunny.natoms(sys.crystal); atol=1e-12)
end
