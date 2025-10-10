module CPU_2D

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2)

using Plots, Printf, Statistics, LinearAlgebra, GeoParams, JLD2

include("Initialize.jl")
include("BC.jl")
include("Geom.jl")
include("Timestep.jl")
include("SaveLoadPlot.jl")
include("PresDense.jl")
include("Rheology.jl")
include("Energy.jl")
include("Solver.jl")

end