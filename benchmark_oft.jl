##
using Pkg, Revise, BenchmarkTools
Pkg.activate(".")
##
import Images
import NPZ: npzread
using GLMakie
using ZStackAnalysis
##
using Base.Threads
nthreads()
##
file_content = npzread("./data/Fed_X63_Z2_SIM.npz")
zstack = Preprocessing.convert_to_float32(file_content["data"])
scale = file_content["scale"]
##
size(zstack)
##

##
xy_region = (1201:1500, 201:500)
actin = interpolate_zstack(zstack[1, :, xy_region...], scale)
actin_adj = Images.adjust_histogram(actin, Images.Equalization(nbins = 256, minval = 0.0, maxval = 0.7))
##
intensity, orientation = LineFilterTransform(actin_adj[1:20, 1:20, 1:20], 10, 7, 20)
@btime OrientationFilterTransform(intensity, orientation, 10, 7, 20)
##
@btime line_segment()