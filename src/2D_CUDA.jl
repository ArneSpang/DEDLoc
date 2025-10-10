module CUDA_2D

using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA, Float64, 2)

using Plots, Printf, Statistics, LinearAlgebra, GeoParams, JLD2

include("InitBC.jl")
include("Geom.jl")
include("Timestep.jl")
include("SaveLoadPlot.jl")
include("Rheology.jl")

end