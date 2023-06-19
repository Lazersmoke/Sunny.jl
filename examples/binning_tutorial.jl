# # Binning Tutorial

using Sunny, GLMakie
import Plots ## Plotting tools as import to avoid conflict with GLMakie

# We specify the crystal lattice structure of CTFD using the lattice parameters specified by
#
# W Wan et al 2020 J. Phys.: Condens. Matter 32 374007 DOI 10.1088/1361-648X/ab757a
latvecs = lattice_vectors(8.113,8.119,12.45,90,100,90)
positions = [[0,0,0]]
types = ["Cu"]
formfactors  = [FormFactor(1,"Cu2")]
xtal = Crystal(latvecs,positions;types);

# We will use a somewhat small periodic lattice size of 5x5x3 in order
# to showcase the effect of a finite lattice size.
latsize = (6,6,4);

# In this system, the magnetic lattice is the same as the chemical lattice,
# and there is a spin-1/2 dipole on each site.
magxtal = xtal;
valS = 1/2;
sys = System(magxtal, latsize, [SpinInfo(1;S = valS)], :dipole; seed=1);

# Quoted value of J = +6.19(2) meV (antiferromagnetic) between
# nearest neighbors on the square lattice
J = 6.19 # meV
characteristic_energy_scale = abs(J * valS)
set_exchange!(sys,J,Bond(1,1,[1,0,0]))
set_exchange!(sys,J,Bond(1,1,[0,1,0]))

# Thermalize the system using the Langevin intergator.
# The timestep and the temperature are roughly based off the characteristic
# energy scale of the problem.
Δt = 0.05 / characteristic_energy_scale
kT = 0.01 * characteristic_energy_scale
langevin = Langevin(Δt; λ=0.1, kT=kT)
randomize_spins!(sys);
for i in 1:10_000 # Long enough to reach equilibrium
    step!(sys, langevin)
end 

# The neutron spectrometer used in the experiment had an incident neutron energy of 36.25 meV.
# Since this is the most amount of energy that can be deposited by the neutron into the sample, we
# don't need to consider energies higher than this.
ωmax = 36.25;

# We choose the resolution in energy (specified by the number of nω modes resolved)
# to be ≈20× better than the experimental resolution in order to demonstrate
# the effect of over-resolving in energy.
nω = 480; 
dsf = DynamicStructureFactor(sys; Δt=Δt, nω=nω, ωmax=ωmax, process_trajectory=:symmetrize); 

# We re-sample from the thermal equilibrium distribution several times to increase our sample size
nsamples = 3
for _ in 1:nsamples
    for _ in 1:8000 
        step!(sys, langevin)
    end
    add_sample!(dsf, sys)
end

# Since the SU(N)NY crystal has only finitely many lattice sites, there are finitely
# many ways for a neutron to scatter off of the sample.
# We can visualize this discreteness by plotting each possible Qx and Qz, for example:
Qx = []
Qz = []
for cell in Sunny.all_cells(sys)
    q = (cell.I .- 1) ./ latsize # q is in R.L.U.
    push!(Qx,q[1])
    push!(Qz,q[3])
end
Plots.scatter(Qx,Qz;xlabel="Qx [r.l.u]",ylabel="Qz [r.l.u.]")

# One way to display the structure factor is to create a histogram with
# one bin centered at each discrete scattering possibility.
params = unit_resolution_binning_parameters(dsf)

# Since this is a 4D histogram,
# it further has to be integrated over two of those directions in order to be displayed.
# Here, we integrate over Qy and Energy:
integrate_axes!(params;axes = [2,4]) # Integrate over Qy (2) and E (4)

# Now that we have parameterized the histogram, we can bin our data.
# The arguments beyond `params` specify which dipole, temperature,
# and atomic form factor corrections should be applied during the intensity calculation.
intensity,counts = intensities_binned(dsf,params,Sunny.DipoleFactor(dsf),kT,formfactors);

# With the data binned, we can now plot it. The axes labels give the bin centers of each bin.
function plot_data(params) #hide
intensity,counts = intensities_binned(dsf,params,Sunny.DipoleFactor(dsf),kT,formfactors) #hide
bin_centers = axes_bincenters(params);
Plots.heatmap(bin_centers[1],bin_centers[3],(intensity[:,1,:,1] ./ counts[:,1,:,1])';xlabel="Qx [r.l.u]",ylabel="Qz [r.l.u]",color=cgrad(:viridis,scale = :lin,rev = false))
Plots.scatter!(Qx,Qz)
end #hide
plot_data(params)#hide

# Note that some bins have no scattering vectors at all when the bin size is made too small:
params = unit_resolution_binning_parameters(dsf)#hide
integrate_axes!(params;axes = [2,4])#hide
params.binwidth[1] /= 1.2
params.binwidth[3] /= 2.5
plot_data(params)#hide

# Conversely, making the bins bigger doesn't break anything, but loses resolution:
params = unit_resolution_binning_parameters(dsf)#hide
integrate_axes!(params;axes = [2,4])#hide
params.binwidth[1] *= 2
params.binwidth[3] *= 2
plot_data(params)#hide

# Recall that while we under-resolved in Q by choosing a small lattice, we 
# over-resolved in energy:
x = []#hide
y = []#hide
for omega = ωs(dsf), qx = unique(Qx)#hide
    push!(x,qx)#hide
    push!(y,omega)#hide
end#hide
Plots.scatter(x,y;xlabel="Qx [r.l.u.]",ylabel="Energy [meV]")#hide

# Let's make a new histogram which includes the energy axis.
# The x-axis of the histogram will be a 1D cut from `Q = [0,0,0]` to `Q = [1,1,0]`
x_axis_bin_count = 10
cut_width = 0.3
params = one_dimensional_cut_binning_parameters(dsf,[0,0,0],[1,1,0],x_axis_bin_count,cut_width)

# There are no longer any scattering vectors exactly in the plane of the cut. Instead,
# as described in the `BinningParameters` output above, the transverse Q
# directions are integrated over, so slightly out of plane points are included.
#
# We plot the intensity on a log-scale to improve visibility.
intensity,counts = intensities_binned(dsf,params,Sunny.DipoleFactor(dsf),kT,formfactors)
bin_centers = axes_bincenters(params);
Plots.heatmap(bin_centers[1],bin_centers[4],log10.(intensity[:,1,1,:] ./ counts[:,1,1,:])';xlabel="Progress along cut [r.l.u]",ylabel="Energy [meV]",color=cgrad(:viridis,scale = :lin,rev = false))

# By reducing the number of energy bins to be closer to the number of bins on the x-axis, we can make the dispersion curve look nicer:
params.binwidth[4] *= 20
intensity,counts = intensities_binned(dsf,params,Sunny.DipoleFactor(dsf),kT,formfactors)#hide
bin_centers = axes_bincenters(params);#hide
Plots.heatmap(bin_centers[1],bin_centers[4],log10.(intensity[:,1,1,:] ./ counts[:,1,1,:])';xlabel="Progress along cut [r.l.u]",ylabel="Energy [meV]",color=cgrad(:viridis,scale = :lin,rev = false))#hide

# Now we will re-run everything at a more reasonable resolution...
latsize = (20,20,3);
## Re-run everything...
sys = System(magxtal, latsize, [SpinInfo(1;S = valS)], :dipole; seed=1);#hide
set_exchange!(sys,J,Bond(1,1,[1,0,0]))#hide
set_exchange!(sys,J,Bond(1,1,[0,1,0]))#hide
randomize_spins!(sys);#hide
for i in 1:10_000#hide
    step!(sys, langevin)#hide
end#hide
nω = 240;#hide
dsf = DynamicStructureFactor(sys; Δt=Δt, nω=nω, ωmax=ωmax, process_trajectory=:symmetrize);#hide
nsamples = 3#hide
for _ in 1:nsamples#hide
    for _ in 1:8000#hide
        step!(sys, langevin)#hide
    end#hide
    add_sample!(dsf, sys)#hide
end#hide

# ... and make a connected path plot.
# Note that the parameters describing each sub-histogram are recorded in `paramsList`.
# These parameters can then be used to bin the experimental data into the same bins
# as are being modelled by Sunny.
points = [[1,   1, 0], 
          [1,   0, 0],
          [1/2, 1/2, 0],
          [1, 1, 0]]
labels = ["($(p[1]),$(p[2]),$(p[3]))" for p in points]
cut_width = 0.3
density = 15
paramsList, markers, ranges = connected_path_bins(dsf,points,density,cut_width;cut_height = 4.0)

total_bins = ranges[end][end]
energy_bins = 24
is = zeros(Float64,total_bins,energy_bins)
for k in 1:length(paramsList)
    paramsList[k].binwidth[4] *= 10
    h,c = intensities_binned(dsf,paramsList[k],Sunny.DipoleFactor(dsf),kT,formfactors)
    is[ranges[k],:] = (h[:,1,1,:] ./ c[:,1,1,:])
end

# In this case, the sub-histograms are generated as follows.
# First cut:
paramsList[1]

# Second cut:
paramsList[2]

# Third cut:
paramsList[3]

# See:
GLMakie.heatmap(1:size(is,1), ωs(dsf), is;
    colorrange=(0.0, 0.2),
    axis = (
        ylabel = "meV",
        xticks = (markers, labels),
        xticklabelrotation=π/8,
        xticklabelsize=12,
    )
)
