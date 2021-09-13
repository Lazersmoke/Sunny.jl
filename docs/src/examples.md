# Examples

The examples stepped through here are available in `examples/` as full loadable files containing executable functions. Specifically, we work through here the simulations
performed in `examples/reproduce_testcases.jl`. All of the plotting code in here
currently depends on an installation of [Plots.jl](http://docs.juliaplots.org/latest/).

The high-level outline of performing a simulation is:

1. Create a [`Crystal`](@ref), either by providing explicit geometry information
    (Example 1), or by loading a `.cif` file (Example 2).
2. Using the `Crystal`, construct a [`Lattice`](@ref) which specifies the system
    size, and a collection of [Interactions](@ref) assembled into a [`Hamiltonian`](@ref).
3. Assemble a [`SpinSystem`](@ref) using the newly created `Lattice` and `Interaction`s.
4. Construct a sampler, either a [`LangevinSampler`](@ref) (Example 1), or a 
    [`MetropolisSampler`](@ref) (Example 2).
5. Use the sampler directly to sample new states, or use it to perform [Structure factor calculations](@ref).

Defining interactions in step (2) can be aided by our utilities for symmetry analysis, demonstrated at the bottom of this page.

In all examples, we will assume that `FastDipole` and `StaticArrays` have been loaded:

```julia
using FastDipole
using StaticArrays
```

## Example 1: Diamond lattice with antiferromagnetic Heisenberg interactions

In this example, we will step through the basic steps needed to set up and run
a spin dynamics simulation, with finite-$T$ statistics obtained by using Langevin
dynamics. The full example is contained in the function `test_diamond_heisenberg_sf()`
within `examples/reproduce_testcases.jl`.

**(1)** We construct a diamond lattice by explicitly defining the lattice geometry. We will use the conventional cubic Bravais lattice with an 8-site basis. Our simulation box
will be ``8 \times 8 \times 8`` unit cells along each axis. Since we're not thinking about a specific system, we label all of the sites with the arbitrary species label `"A"`.

```julia
lat_vecs = [4.0 0.0 0.0;
            0.0 4.0 0.0;
            0.0 0.0 4.0]
basis_vecs = [
    [0.0, 0.0, 0.0],
    [0.0, 1/2, 1/2],
    [1/2, 0.0, 1/2],
    [1/2, 1/2, 0.0],
    [3/4, 3/4, 3/4],
    [3/4, 1/4, 1/4],
    [1/4, 3/4, 1/4],
    [1/4, 1/4, 3/4]
]
basis_labels = fill("A", 8)
latsize = [8, 8, 8]
lattice = Lattice{3}(lat_vecs, basis_vecs, basis_labels, latsize)
crystal = Crystal(lattice)
```

Here, `Lattice` is a type which holds the geometry of our simulation box. This type can be indexed into as an array of size `B×Lx×Ly×Lz`, with the first index selecting the sublattice and the remaining three selecting the unit cell, and the absolute position of the selected site will be given. For example, the following gives the position of the second sublattice site in the unit cell which is fifth along each axis: 

```
julia> lattice[2, 5, 5, 5]
3-element SVector{3, Float64} with indices SOneTo(3):
 20.0
 22.0
 22.0
```

(To save a minor amount of arithmetic, the bottom-left corner of the simulation box lives at ``a+b+c`` rather than 0). From this `Lattice`, we created a `Crystal` that infers extra information about the symmetries of the lattice and is needed to define interactions in our system later.

**(2)** In step 1, we ended up already creating our `Lattice`, so all that is left is
to define our Heisenberg interactions. We want to set up nearest-neighbor antiferromagnetic interactions with a strength of ``J = 28.28~\mathrm{K}``. One nearest-neighbor bond is the one connecting basis site 3 with basis site 6 within a single unit cell. (We can figure this out using our tools for symmetry analysis, at the bottom of this page).

```julia
J = 28.28           # Units of K
interactions = [
    Heisenberg(J, crystal, Bond{3}(3, 6, SA[0,0,0])),
]
ℋ = Hamiltonian{3}(interactions)
```
Here, the `3` in both `Bond{3}` and `Hamiltonian{3}` indicates that they are defined in the context of a 3-dimensional system.

**(3)** Assembling a `SpinSystem` is straightforward. Then, we will randomize the system so that all spins are randomly placed on the unit sphere.

```julia
sys = SpinSystem(lattice, ℋ)
rand!(sys)
```

The `SpinSystem` type is the central type used throughout our simulations. Internally, it contains the spin degrees of freedom which evolve during the simulation as well as the Hamiltonian defining how this evolution should occur. The type is indexable in the same way as `Lattice`, by providing a sublattice index alongside indices into the unit cell axes:

```
sys[2, 5, 5, 5]
3-element SVector{3, Float64} with indices SOneTo(3):
  0.22787294226659
  0.41462045511988654
 -0.8810015893169237
```

Now, we are obtaining the actual spin variable living at site `(2, 5, 5, 5)`.

**(4)** We will simulate this system using Langevin dynamics, so we need to create a [`LangevinSampler`](@ref). Note that the units of integration time and temperature are relative to the units implicitly used when setting up the interactions.

```julia
Δt = 0.02 / J       # Units of 1/K
kT = 4.             # Units of K
α  = 0.1
kB = 8.61733e-5     # Units of eV/K
nsteps = 20000
sampler = LangevinSampler(sys, kT, α, Δt, nsteps)
```

At this point we can call `sample!(sampler)` to produce new samples of the system, which will be reflected in the state of `sysc`. Instead, we will proceed to calculate
the finite-$T$ structure factor using our built-in routines.

**(5)** The full process of calculating a structure factor is handled
by [`structure_factor`](@ref). Internally, this function:

1. Thermalizes the system for a while
2. Samples a new thermal spin configuration
3. Performs constant-energy LL dynamics to obtain a Fourier-transformed
    dynamics trajectory. Use this trajectory to calculate a structure
    factor contribution ``S^{\alpha,\beta}(\boldsymbol{q}, \omega)``.
4. Repeat steps (2,3), averaging structure factors across samples.

See the documentation of [`structure_factor`](@ref) for details of how
this process is controlled by the function arguments, and how to properly
index into the resulting array.

In this example, we will just look at the diagonal elements of this
matrix along some cuts in reciprocal space. To improve statistics,
we average these elements across the ``x, y, z`` spin directions since
they are all symmetry equivalent in this model.

```julia
meas_rate = 10
S = structure_factor(
    sys, sampler; num_samples=5, dynΔt=Δt, meas_rate=meas_rate,
    num_freqs=1600, bz_size=(1,1,2), therm_samples=10, verbose=true
)

# Retain just the diagonal elements, which we will average across the
#  symmetry-equivalent directions.
avgS = zeros(Float64, axes(S)[3:end])
for α in 1:3
    @. avgS += real(S[α, α, :, :, :, :])
end

```

We then plot some cuts using a function `plot_many_cuts` defined within
the example script. (I.e. this code block will not successfully execute unless
you `include("examples/reproduce_testcases.jl)`). We omit this code here as it's just
a large amount of indexing and plotting code, but for details see the script.

```julia
# Calculate the maximum ω present in our FFT
# Need to scale by (S+1) with S=3/2 to match the reference,
#  and then convert to meV.
maxω = 1000 * 2π / ((meas_rate * Δt) / kB) / (5/2)
p = plot_many_cuts(avgS; maxω=maxω, chopω=5.0)
display(p)
```

## Example 2: FeI₂ with a complex collection of interactions

In this example, we work through performing a more complicated and realistic
simulation of FeI₂. While the number of interactions is much larger, the general
process will be remarkably similar. We will also see how to perform sampling using
Metropolis Monte Carlo through the [`MetropolisSampler`](@ref) type. The full example is
contained in the function `test_FeI2_MC()` within `examples/reproduce_testcases.jl`.

**(1)** As before, the first step is to make a [`Crystal`](@ref). However, this time
we will load the crystal directly from a common crystallographic file format
called a `.cif`. You can download the structure file from [this link](https://materials.springer.com/isp/crystallographic/docs/sd_0548497). Then, we can load it as:

```julia
cryst = Crystal("./FeI2.cif")
cryst = subcrystal(cryst, "Fe2+")
```

(Be sure to change `"./FeI2.cif"` to whatever filename you've locally saved the file as.)

As only the Fe atoms are spinful, the second line here is selecting out just them.
However, the [`subcrystal`](@ref) function critically retains information about
the symmetry of the crystal structure with the I atoms present, which is
important for symmetry-constraining allowed interactions between sites.

**(2)** We proceed to define our Hamiltonian similarly as before, however
this time many more interactions are present. See the documentation on the
[Interactions](@ref) for extended descriptions of each.

```julia

# All units in meV
J1mat = [-0.397  0      0    ;
          0     -0.075 -0.261;
          0     -0.261 -0.236]
J1 = GeneralCoupling(J1mat, cryst, Bond{3}(1, 1, [1, 0, 0]), "J1")
J2 = DiagonalCoupling([0.026, 0.026, 0.113], cryst, Bond{3}(1, 1, [1, -1, 0]), "J2")
J3 = DiagonalCoupling([0.166, 0.166, 0.211], cryst, Bond{3}(1, 1, [2, 0, 0]), "J3")
J0′ = DiagonalCoupling([0.037, 0.037, -0.036], cryst, Bond{3}(1, 1, [0, 0, 1]), "J0′")
J1′ = DiagonalCoupling([0.013, 0.013, 0.051], cryst, Bond{3}(1, 1, [1, 0, 1]), "J1′")
J2a′ = DiagonalCoupling([0.068, 0.068, 0.073], cryst, Bond{3}(1, 1, [1, -1, 1]), "J2a′")

D = OnSite([0.0, 0.0, -2.165/2], "D")

ℋ = Hamiltonian{3}([J1, J2, J3, J0′, J1′, J2a′, D])
```

Using our `Crystal`, we also need to generate a `Lattice` of some size to run our
simulation on. In this example, we'll work with a modestly large system of size
``16\times 20\times 4`` along the ``(a, b, c)`` axes. We choose to make the ``a``
and ``b`` lengths different to artifically break a sixfold symmetry present in
the system to help the Monte Carlo find the correct ground state later on.

```julia
lattice = Lattice(cryst, (16, 20, 4))
```

To get better insight into the geometry and the long set of pair interactions
we've defined above, we can take a look at both using the following plotting
function (you may want to replace `lattice` with something smaller, say ``5 \times 5 \times 3`` to make the bonds easier to see, or adjust `markersize` to make the atoms easier to see):

```julia
plot_bonds(lattice, ℋ; markersize=500)
```

**(3)** As with the previous example, the next step is to make a `SpinSystem` and
randomize it:

```julia
system = SpinSystem(lattice, ℋ)
rand!(system)
```

**(4)** In this example, we'll choose to work with Metropolis Monte Carlo rather
than Langevin sampling. This is necessary in this system due to a very
strong on-site anisotropy (the `OnSite` term) making the spins nearly
Ising-like. Continuous Langevin dyanmics can have ergodicity issues
in these situations, so we have to turn back to the standard Metropolis
randomized spin flip proposals.

```julia
kB = 8.61733e-2  # Boltzmann constant, units of meV/K
kT = 1.0 * kB    # Target simulation temp, in units of meV

sampler = MetropolisSampler(system, kT, 1000)
```

`MetropolisSampler` provides a very similar interface to `LangevinSampler`. Calling
`sample!(sampler)` will perform some number of spin-flip proposals, then return with
`system` updated to a new set of spin values. The `1000` in our constructor is asking
the sampler to perform 1000 sweeps through the system before the `sample!` function
should return.

**(5)**
As in the previous example, we are going to end with computing a dynamic structure
factor tensor using the `structure_factor` function. A heuristic for choosing a
reasonable value of `Δt` using in the Landau-Lifshitz dynamics is `0.01` divided
by the largest energy scale present in the system. Here, that is the on-site
anisotropy with a strength of `2.165/2 ` meV.

To make sure we don't do more work than really necessary, we set how
often `structure_factor` internally stores snapshots (`meas_rate`) to
target a maximum frequency of `target_max_ω`. We also will only collect
the structure factor along two Brillouin zones along the first reciprocal axis,
by passing `bz_size=(2,0,0)`

The following block of code takes about five minutes on
a test desktop, but if it's taking too long you can
reduce the time either by reducing the number of sweeps
`MetropolisSampler` does, or the `num_samples` or
`num_freqs` in the structure factor computation.

```julia
Δt = 0.01 / (2.165/2)       # Units of 1/meV
# Highest energy/frequency we actually care about resolving
target_max_ω = 10.          # Units of meV
# Interval number of steps of dynamics before collecting a snapshot for FFTs
meas_rate = convert(Int, div(2π, (2 * target_max_ω * Δt)))

sampler = MetropolisSampler(system, kT, 500)
println("Starting structure factor measurement...")
S = structure_factor(
    system, sampler; num_samples=15, meas_rate=meas_rate,
    num_freqs=1000, bz_size=(2,0,0), verbose=true, therm_samples=15
)
```

Given the full complex-valued ``\mathcal{S}^{\alpha \beta}(\boldsymbol{q}, \omega)``,
we can reduce it to the real-value experimentally-observable cross section by projection
each `\mathcal{S}^{\alpha \beta}` using the neutron dipole factor. See the
[`dipole_factor`](@ref) documentation for more details. (To be truly comparable
to experiment, a few more steps of processing need to be done which are currently
unimplemented.)

```julia
S = dipole_factor(S, lattice)
```

(Will add info here about plotting when better structure factor plotting functions
are implemented.)

In the following example, we'll take a closer look at how to make
some more manual measurements of the system.

## Example 3: Making manual measurements within a Monte Carlo simulation

In this example, we will perform an extended Monte Carlo simulation of the
same system as in the previous example, but will perform a careful
thermal annealing down to low temperatures and measure an ``E(T)`` curve
along the way. To do so, we will need to use the sampling tools a bit
more manually.

As we're using the same system as before, the setup will be identical. The lines
are copied below for convenience, but see the previous example for an
explanation of each step.

```julia
cryst = Crystal("./FeI2.cif")
cryst = subcrystal(cryst, "Fe2+")

# All units in meV
J1mat = [-0.397  0      0    ;
          0     -0.075 -0.261;
          0     -0.261 -0.236]
J1 = GeneralCoupling(J1mat, cryst, Bond{3}(1, 1, [1, 0, 0]), "J1")
J2 = DiagonalCoupling([0.026, 0.026, 0.113], cryst, Bond{3}(1, 1, [1, -1, 0]), "J2")
J3 = DiagonalCoupling([0.166, 0.166, 0.211], cryst, Bond{3}(1, 1, [2, 0, 0]), "J3")
J0′ = DiagonalCoupling([0.037, 0.037, -0.036], cryst, Bond{3}(1, 1, [0, 0, 1]), "J0′")
J1′ = DiagonalCoupling([0.013, 0.013, 0.051], cryst, Bond{3}(1, 1, [1, 0, 1]), "J1′")
J2a′ = DiagonalCoupling([0.068, 0.068, 0.073], cryst, Bond{3}(1, 1, [1, -1, 1]), "J2a′")

D = OnSite([0.0, 0.0, -2.165/2], "D")

ℋ = Hamiltonian{3}([J1, J2, J3, J0′, J1′, J2a′, D])

lattice = Lattice(cryst, (16, 20, 4))

system = SpinSystem(lattice, ℋ)
rand!(system)

sampler = MetropolisSampler(system, 1.0, 10)
```

Now, our goal in the following is to measure an entire ``E(T)`` curve, down
to relatively low temperatures. To help the system find the ground state
correctly at low temperatures, we will use the same system throughout and
slowly "anneal" the temperature from the highest value down to the lowest.

These next few lines are pure Julia which simply sets up the temperatures
we want to measure along, and initializes some `Vector`'s to store some
data during the simulations.

```julia
kB = 8.61733e-2             # Boltzmann constant, units of meV/K

# Units of Kelvin, matching Xiaojian's range
temps = 10 .^ (range(log10(50), stop=0, length=50))
temps_meV = kB .* temps
energies = Float64[]
energy_errors = Float64[]
```

We've chosen to measure along a logarithmic temperature grid spanning ``T \in [1, 50]``,
so that we pack the grid points tighter at lower temperatures where interesting
things occur. `energies` and `energy_errors` are going to hold our measurements 
of the mean energy and the errors at each temperature.

Now, we're going to loop over these temperatures (moving from higher to lower temperatures). At each temperature,
we're going to:

1. Set the temperature of the sampler to the new temperature using `set_temp!`.
2. Thermalize at the new temperature for a while before collecting
    measurements using `thermalize!`.
3. Sample the system 1000 times, and measure the energy of each spin
    configuration. We'll record all of these energies in `temp_energies`.
4. Compute the mean energy and its standard error from our 1000 measurements
5. Push this mean energy and standard error to our `energies` and 
    `energy_errors` vectors.

For simplicity, here we're just going to use the standard error across
all energy measurements as the error. See the `binned_statistics`
function in `examples/reproduce_testcases.jl` to see how to
measure the error more carefully.

The following block of code takes a few minutes to execute. Feel free to sample a sparser temperature grid, play around with some of the thermalization parameters, or perform fewer measurements to try to get it to execute faster.

```julia
using Statistics

for (i, temp) in enumerate(temps_meV)
    println("Temperature $i = $(temp)")

    temp_energies = Float64[]
    set_temp!(sampler, temp)
    thermalize!(sampler, 100)
    for _ in 1:1000
        sample!(sampler) 
        push!(temp_energies, energy(sampler))
    end
    meanE = mean(temp_energies)
    errE  = std(temp_energies) / sqrt(length(temp_energies))
    push!(energies, meanE)
    push!(energy_errors, errE)
end

# Convert energies into energy / spin, in units of K
energies ./= (length(system) * kB)
energy_errors ./= (length(system) * kB)
```

Now, we can plot what we've gotten! If you have the Plots.jl library installed you can do this as:

```julia
using Plots

p = plot(temps, energies, yerror=energy_errors, marker=:true, ms=3, label="Monte Carlo Results")
xlabel!(L"$T$ [K]")
ylabel!(L"$E$ [K]")
p
```

If all has gone well, you should get a plot that looks
something like the following:

![FeI₂ Energy Curve](assets/FeI2_ETcurve.png)

We can take a look at the final low-energy spin configuration
by:

```julia
plot_spins(system; arrowsize=1.5, arrowlength=3, linewidth=0.5)
```

You should see antiferromagnetic stripes within each
``c``-plane, which shift by one lattice site as you
move up each plane!


## Symmetry analysis

When defining pair interactions, we are always defining the interactions on
entire symmetry classes at the same time. To do this, we need to provide the
exchange matrix ``J`` on a specific reference `Bond`, which is then automatically
propagated to all symmetry-equivalent bonds. However, on any given bond, the
exchange matrix must live within a restricted space of ``3 \times 3`` matrices
that is confined by the symmetry properties of the underlying crystal.

To discover all symmetry classes of bonds up to a certain distance while simultaneously learning what the allowed form of the `J` matrix is, construct a `Crystal` then call the function [`print_bond_table`](@ref).

```
julia> lattice = FastDipole.diamond_conventional(1.0, (8, 8, 8))
julia> crystal = Crystal(lattice)
julia> print_bond_table(crystal, 4.0)

Bond{3}(1, 1, [0, 0, 0])
Distance 0, multiplicity 1
Connects [0, 0, 0] to [0, 0, 0]
Allowed coupling:  |A 0 0 |
                   |0 A 0 |
                   |0 0 A |

Bond{3}(3, 6, [0, 0, 0])
Distance 1.732, multiplicity 4
Connects [0.5, 0, 0.5] to [0.75, 0.25, 0.25]
Allowed coupling:  | A  B -B |
                   | B  A -B |
                   |-B -B  A |

Bond{3}(1, 2, [0, 0, 0])
Distance 2.828, multiplicity 12
Connects [0, 0, 0] to [0, 0.5, 0.5]
Allowed coupling:  | B  D  D |
                   |-D  C  A |
                   |-D  A  C |

Bond{3}(1, 6, [0, 0, 0])
Distance 3.317, multiplicity 12
Connects [0, 0, 0] to [0.75, 0.25, 0.25]
Allowed coupling:  |B C C |
                   |C D A |
                   |C A D |

Bond{3}(1, 1, [1, 0, 0])
Distance 4, multiplicity 6
Connects [0, 0, 0] to [1, 0, 0]
Allowed coupling:  |A 0 0 |
                   |0 B 0 |
                   |0 0 B |
```

Each block represents one symmetry equivalence class of bonds, along with a single
representative ("canonical") `Bond` for that class and the allowed exchange coupling
matrix on that canonical bond.

You can also query what the allowed exchange matrix is on a specific bond using [`allowed_J`](@ref).

```
julia> allowed_J(crystal, Bond{3}(1, 5, [1,-1,0]))

3×3 Matrix{String}:
 "D"  "A"  "B"
 "A"  "E"  "C"
 "B"  "C"  "F"
```