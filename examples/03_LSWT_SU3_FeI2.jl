# # 3. Multi-flavor spin wave simulations of FeI₂
# 
# This tutorial illustrates various powerful features in Sunny, including
# symmetry analysis, energy minimization, and spin wave theory with multi-flavor
# bosons.
#
# Our context will be FeI₂, an effective spin-1 material with strong single-ion
# anisotropy. Quadrupolar fluctuations give rise to a single-ion bound state
# that is observable in neutron scattering, and cannot be described by a
# dipole-only model. This tutorial illustrates how to use the linear spin wave
# theory of SU(3) coherent states (i.e. 2-flavor bosons) to model the magnetic
# spectrum of FeI₂. The original study was performed in [Bai et al., Nature
# Physics 17, 467–472 (2021)](https://doi.org/10.1038/s41567-020-01110-1).
#
# ```@raw html
# <img src="https://raw.githubusercontent.com/SunnySuite/Sunny.jl/main/docs/src/assets/FeI2_crystal.jpg" style="float: left;" width="400">
# ```
#
# The Fe atoms are arranged in stacked triangular layers. The effective spin
# Hamiltonian takes the form,
# 
# ```math
# \mathcal{H}=\sum_{(i,j)} 𝐒_i ⋅ J_{ij} 𝐒_j - D\sum_i \left(S_i^z\right)^2,
# ```
#
# where the exchange matrices ``J_{ij}`` between bonded sites ``(i,j)`` include
# competing ferromagnetic and antiferromagnetic interactions. This model also
# includes a strong easy axis anisotropy, ``D > 0``.

# Load packages.

using Sunny, GLMakie

# Construct the chemical cell of FeI₂ by specifying the lattice vectors and the
# full set of atoms.

units = Units(:meV, :angstrom)
a = b = 4.05012  # Lattice constants for triangular lattice (Å)
c = 6.75214      # Spacing between layers (Å)
latvecs = lattice_vectors(a, b, c, 90, 90, 120)
positions = [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]
types = ["Fe", "I", "I"]
cryst = Crystal(latvecs, positions; types)

# Observe that Sunny inferred the space group, 'P -3 m 1' (164) and labeled the
# atoms according to their point group symmetries. Only the Fe atoms are
# magnetic, so we focus on them with [`subcrystal`](@ref). Importantly, this
# operation preserves the spacegroup symmetries.

cryst = subcrystal(cryst, "Fe")
view_crystal(cryst)

# ### Symmetry analysis
#
# The command [`print_symmetry_table`](@ref) provides a list of all the
# symmetry-allowed interactions out to 8 Å.

print_symmetry_table(cryst, 8.0)

# The allowed ``g``-tensor is expressed as a 3×3 matrix in the free coefficients
# `A`, `B`, ... The allowed single-ion anisotropy is expressed as a linear
# combination of Stevens operators. The latter correspond to polynomials of the
# spin operators, as we will describe below.
# 
# The allowed exchange interactions are given as 3×3 matrices for representative
# bonds. The notation `Bond(i, j, n)` indicates a bond between atom indices `i`
# and `j`, with cell offset `n`. Note that the order of the pair ``(i, j)`` is
# significant if the exchange tensor contains antisymmetric
# Dzyaloshinskii–Moriya (DM) interactions.
# 
# The bonds can be visualized in the `view_crystal` interface. By default,
# `Bond(1, 1, [1,0,0])` is toggled on, to show the 6 nearest-neighbor Fe-Fe
# bonds on a triangular lattice layer. Toggling `Bond(1, 1, [0,0,1])` shows the
# Fe-Fe bond between layers, etc.

# ### Defining the spin model

# Construct a system [`System`](@ref) with spin-1 and g=2 for the Fe ions.
#
# Recall that a quantum spin-1 state is, in general, a linear combination of
# basis states ``|m⟩`` with well-defined angular momentum ``m = -1, 0, 1``. FeI₂
# has a strong easy-axis anisotropy, which stabilizes a single-ion bound state
# of zero angular momentum, ``|m=0⟩``. Such a bound state is inaccessible to
# traditional spin wave theory, which works with dipole expectation values of
# fixed magnitude. This physics can, however, be captured with a theory of
# SU(_N_) coherent states, where ``N = 2S+1 = 3`` is the number of levels. We
# will therefore select `:SUN` mode instead of `:dipole` mode.
# 
# Selecting an optional random number `seed` will make the calculations exactly
# reproducible.

sys = System(cryst, [SpinInfo(1, S=1, g=2)], :SUN, seed=2)

# Set the exchange interactions for FeI₂ following the fits of [Bai et
# al.](https://doi.org/10.1038/s41567-020-01110-1)

J1pm   = -0.236 # (meV)
J1pmpm = -0.161
J1zpm  = -0.261
J2pm   = 0.026
J3pm   = 0.166
J′0pm  = 0.037
J′1pm  = 0.013
J′2apm = 0.068

J1zz   = -0.236
J2zz   = 0.113
J3zz   = 0.211
J′0zz  = -0.036
J′1zz  = 0.051
J′2azz = 0.073

J1xx = J1pm + J1pmpm 
J1yy = J1pm - J1pmpm
J1yz = J1zpm

set_exchange!(sys, [J1xx   0.0    0.0;
                    0.0    J1yy   J1yz;
                    0.0    J1yz   J1zz], Bond(1,1,[1,0,0]))
set_exchange!(sys, [J2pm   0.0    0.0;
                    0.0    J2pm   0.0;
                    0.0    0.0    J2zz], Bond(1,1,[1,2,0]))
set_exchange!(sys, [J3pm   0.0    0.0;
                    0.0    J3pm   0.0;
                    0.0    0.0    J3zz], Bond(1,1,[2,0,0]))
set_exchange!(sys, [J′0pm  0.0    0.0;
                    0.0    J′0pm  0.0;
                    0.0    0.0    J′0zz], Bond(1,1,[0,0,1]))
set_exchange!(sys, [J′1pm  0.0    0.0;
                    0.0    J′1pm  0.0;
                    0.0    0.0    J′1zz], Bond(1,1,[1,0,1]))
set_exchange!(sys, [J′2apm 0.0    0.0;
                    0.0    J′2apm 0.0;
                    0.0    0.0    J′2azz], Bond(1,1,[1,2,1]))

# The function [`set_onsite_coupling!`](@ref) assigns a single-ion anisotropy.
# The argument can be constructed using [`spin_matrices`](@ref) or
# [`stevens_matrices`](@ref). Here we use Julia's anonymous function syntax to
# assign an easy-axis anisotropy along the direction ``\hat{z}``.

D = 2.165 # (meV)
set_onsite_coupling!(sys, S -> -D*S[3]^2, 1)

# ### Finding the ground state

# This model has been carefully designed so that energy minimization produces
# the physically correct magnetic ordering. Using [`set_dipole!`](@ref), this
# magnetic structure can be entered manually. Sunny also provides tools to
# search for an unknown magnetic order, as we will now demonstrate.
#
# To minimize bias in the search, use [`resize_supercell`](@ref) to create a
# relatively large system of 4×4×4 chemical cells. Randomize all spins (as SU(3)
# coherent states) and minimize the energy.

sys = resize_supercell(sys, (4, 4, 4))
randomize_spins!(sys)
minimize_energy!(sys)

# A positive number above indicates that the procedure has converged to a local
# energy minimum. The configuration, however, may still have defects. This can
# be checked by visualizing the expected spin dipoles, colored according to
# their ``z``-components.

plot_spins(sys; color=[s[3] for s in sys.dipoles])

# To better understand the spin configuration, we could inspect the static
# structure factor ``\mathcal{S}(𝐪)`` in the 3D space of momenta ``𝐪``. For
# this, Sunny provides [`SampledCorrelationsStatic`](@ref). Here, however, we
# will use [`print_wrapped_intensities`](@ref), which gives static intensities
# averaged over the individual Bravais sublattices (in effect, all ``𝐪``
# intensities are periodically wrapped to the first Brillouin zone).

print_wrapped_intensities(sys)

# The known zero-field energy-minimizing magnetic structure of FeI₂ is a two-up,
# two-down order. It can be described as a generalized spiral with a single
# propagation wavevector ``𝐤``. Rotational symmetry allows for three equivalent
# orientations: ``±𝐤 = [0, -1/4, 1/4]``, ``[1/4, 0, 1/4]``, or
# ``[-1/4,1/4,1/4]``. Small systems can spontaneously break this symmetry, but
# for larger systems, defects and competing domains are to be expected.
# Nonetheless, `print_wrapped_intensities` shows large intensities consistent
# with a subset of the known ordering wavevectors.
#
# Let's break the three-fold symmetry by hand. The function
# [`suggest_magnetic_supercell`](@ref) takes one or more ``𝐤`` modes, and
# suggests a magnetic cell shape that is commensurate.

suggest_magnetic_supercell([[0, -1/4, 1/4]])

# Calling [`reshape_supercell`](@ref) yields a much smaller system, making it
# much easier to find the global energy minimum. Plot the system again, now
# including "ghost" spins out to 12Å, to verify that the magnetic order is
# consistent with FeI₂.

sys_min = reshape_supercell(sys, [1 0 0; 0 2 1; 0 -2 1])
randomize_spins!(sys_min)
minimize_energy!(sys_min);
plot_spins(sys_min; color=[s[3] for s in sys_min.dipoles], ghost_radius=12)

# ### Spin wave theory
#
# Now that the system has been relaxed to an energy minimized ground state, we
# can calculate the spin wave spectrum. Because we are working with a system of
# SU(3) coherent states, this calculation will require a multi-flavor boson
# generalization of the usual spin wave theory.

# Calculate and plot the spectrum along a momentum-space path that connects a
# sequence of high-symmetry ``𝐪``-points. These Sunny commands are identical to
# those described in [`our previous CoRh₂O₄ tutorial`](@ref "1. Spin wave
# simulations of CoRh₂O₄").

qs = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]]
path = q_space_path(cryst, qs, 500)
swt = SpinWaveTheory(sys_min; measure=ssf_perp(sys_min))
res = intensities_bands(swt, path)
plot_intensities(res; units)

# To make direct comparison with inelastic neutron scattering (INS) data, we
# must account for empirical broadening of the data. Model this using a
# [`lorentzian`](@ref) kernel, with a full-width at half-maximum of 0.3 meV.

kernel = lorentzian(fwhm=0.3)
energies = range(0, 10, 300);  # 0 < ω < 10 (meV)

# Also, a real FeI₂ sample will exhibit competing magnetic domains. We use
# [`domain_average`](@ref) to average over the three possible domain
# orientations. This involves 120° rotations about the axis ``\hat{z} = [0, 0,
# 1]`` in global Cartesian coordinates.

rotations = [([0,0,1], n*(2π/3)) for n in 0:2]
weights = [1, 1, 1]
res = domain_average(cryst, path; rotations, weights) do path_rotated
    intensities(swt, path_rotated; energies, kernel)
end
plot_intensities(res; units, colormap=:viridis)

# This result can be directly compared to experimental neutron scattering data
# from [Bai et al.](https://doi.org/10.1038/s41567-020-01110-1)
# ```@raw html
# <img src="https://raw.githubusercontent.com/SunnySuite/Sunny.jl/main/docs/src/assets/FeI2_intensity.jpg">
# ```
#
# (The publication figure used a non-standard coordinate system to label the
# wave vectors.)
# 
# To get this agreement, the theory of SU(3) coherent states is essential. The
# lower band has large quadrupolar character, and arises from the strong
# easy-axis anisotropy of FeI₂.
#
# An interesting exercise is to repeat the same study, but using `:dipole` mode
# instead of `:SUN`. That alternative choice would constrain the coherent state
# dynamics to the space of dipoles only, and the flat band of single-ion bound
# states would be missing.