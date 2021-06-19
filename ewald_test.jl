using Revise
using TOML
using Profile

includet("Lattice.jl")
includet("Systems.jl")
includet("Ewald.jl")

config = TOML.tryparsefile("cubic.toml")
lat = _parse_lattice(config["lattice"])

sys = ChargeSystem(lat)
randn!(sys)

# Ready for ewald_sum_monopole(sys)
# ewald_sum_monopole(sys)
# Profile.clear_malloc_data()
# ewald_sum_monopole(sys)

### Notes:

"""
This function allocates a ton of memory:

function mm(sys::ChargeSystem{D}) where{D}
    v = zeros(MVector{3})
    accum = zeros(MVector{3})
    M = sys.lattice.lat_vecs
    for jkl in CartesianIndices((25, 25, 25))
        jkl = convert(SVector, jkl)
        mul!(v, M, jkl)         <---- Allocations!
        accum .+= v
    end
    return accum
end

BUT, this one doesn't:

function mm(sys::ChargeSystem{D}) where{D}
    v = zeros(MVector{3})
    accum = zeros(MVector{3})
    M = @SMatrix randn(3, 3)
    for jkl in CartesianIndices((25, 25, 25))
        jkl = convert(SVector, jkl)
        mul!(v, M, jkl)         
        accum .+= v
    end
    return accum
end

This one also does not, no matter if I pass in a fixed M,
 or pass in sys.lattice.lat_vecs. ???

function mm(M::SMatrix)
   v = zeros(MVector{3})
   accum = zeros(MVector{3})
   for jkl in CartesianIndices((25, 25, 25))
       jkl = convert(SVector, jkl)
       mul!(v, M, jkl)
       accum .+= v
   end
   return accum
end



@code_typed on the mul! line of first version returns an Any:

│    %14  = StaticArrays.Size(%4)::Size{(3, 3)}
│           StaticArrays._mul!($(QuoteNode(StaticArrays.TSize{(3,), :any}())), %1, %14, $(QuoteNode(Size(3,))), %4, %13, $(QuoteNode(StaticArrays.NoMulAdd{Float64, Float64}())))::Any

while the second is well-typed

 %13  = StaticArrays._mul!::typeof(StaticArrays._mul!)
│           invoke %13($(QuoteNode(StaticArrays.TSize{(3,), :any}()))::StaticArrays.TSize{(3,), :any}, %1::MVector{3, Float64}, $(QuoteNode(Size(3, 3)))::Size{(3, 3)}, $(QuoteNode(Size(3,)))::Size{(3,)}, %3::SMatrix{3, 3, Float64, 9}, %12::SVector{3, Int64}, $(QuoteNode(StaticArrays.NoMulAdd{Float64, Float64}()))::StaticArrays.NoMulAdd{Float64, Float64}, $(QuoteNode(Val{1}()))::Val{1})::MVector{3, Float64}

Is it because of the final L type of the SMatrix? This should be inferrable...

Base.getfield(%3, :lat_vecs)::SMatrix{3, 3, Float64, L} where L

YES, it is this. Have to change Lattice{D} -> Lattice{D, L}...

"""