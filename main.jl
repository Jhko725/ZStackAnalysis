##
using Pkg, Revise, BenchmarkTools
Pkg.activate(".")
## 
import Images
import NPZ: npzread
using GLMakie
using ZStackAnalysis
##
file_content = npzread("./Fed_X63_Z2_SIM.npz")
#file_content = npzread("./SIM_Jing.npz")
function convert_to_float32(zstack::AbstractArray)
    return Float32.(reinterpret(Images.N0f16, zstack))
end
zstack = convert_to_float32(file_content["data"])
scale = file_content["scale"]
##
size(zstack)
##
using FileIO
jing = load("./Image 6_Out crop.tif")
##
jing_reshaped = reshape(jing, 2560, 2560, 4, 21)
size(jing_reshaped)
##
size(zstack)
##

image(jing_reshaped[:, :, 3, 20])
##
jing_overlay = Float32.(mean(jing_reshaped, dims = 4)[1000:1500, 1000:1500, :, 1])
image(Images.adjust_histogram(jing_overlay[:, :,3], Images.Equalization(nbins = 256, minval = 0.0, maxval = 0.5)))
##
typeof(jing_overlay)

##
image(@view zstack[2, 1, :, :])
##
vol = zstack[1, :, 1201:1500, 201:500]
image(vol[1, :, :])
##


##
xy_region = (1201:1500, 201:500)
actin = interpolate_zstack(zstack[1, :, xy_region...], scale)
desmin = interpolate_zstack(zstack[2, :, xy_region...], scale)
##
jing3D = interpolate_zstack(Float32.(jing_reshaped[1000:1500, 1000:1500, :, :]), [3.13e-8, 3.13e-8, 1.1e-7])
size(jing3D)
##
volume(actin, algorithm = :mip, colormap = :Greens_9, transparancy = true)
#volume(desmin, algorithm = :mip, colormap = :Reds_9, transparancy = true)
##
actin_adj = Images.adjust_histogram(actin, Images.Equalization(nbins = 256, minval = 0.0, maxval = 0.7))
desmin_adj = Images.adjust_histogram(desmin, Images.Equalization(nbins = 256, minval = 0.0, maxval = 0.7))
##
#figure = Figure()
#axis3 = Axis3(figure[1, 1])
volume(actin_adj, algorithm = :mip, colormap = :Greens_9, transparancy = true)
volume(desmin_adj, algorithm = :mip, colormap = :Reds_9, transparancy = true)
current_figure()
##
size(actin_adj)
##

vol_RGB = Images.colorview(Images.RGB, desmin_adj, actin_adj, Images.zeroarray)
z_ind = 120
image(vol_RGB[z_ind, :, :])
##
using ZStackAnalysis
##
actin_I, actin_θ = LineFilterTransform(actin_adj[z_ind, :, :], 10, 7, 20)
desmin_I, desmin_θ = LineFilterTransform(desmin_adj[z_ind, :, :], 10, 7, 20)
actin_oft = OrientationFilterTransform(actin_I, actin_θ, 10, 7, 20)
desmin_oft = OrientationFilterTransform(desmin_I, desmin_θ, 10, 7, 20)
##
jing_I, jing_θ = LineFilterTransform(jing_overlay[:, :, 3], 10, 15, 20)
##
jing_oft = OrientationFilterTransform(jing_I, jing_θ, 10, 15, 20)
##
angle_to_color(angle) = Images.colorsigned(Images.RGB(1.0, 0.0, 0.0), Images.RGB(0.5, 0.0, 0.5), Images.RGB(0.0, 0.0, 1.0))(Images.scalesigned(π/2)(angle))
##
fig = Figure(resolution = (1500, 1000))
ax1 = fig[1, 1] = GLMakie.Axis(fig, title = "Raw Image")
img1 = image!(ax1, jing_overlay[:, :, 1])
ax2 = fig[2, 1] = GLMakie.Axis(fig, title = "Line Filter Transform (intensity)")
img2 = image!(ax2, Images.scaleminmax(0, maximum(jing_I)).(jing_I))
ax3 = fig[2, 2] = GLMakie.Axis(fig, title = "Line Filter Transform (orientation)")

img3 = image!(ax3, angle_to_color.(jing_θ))
ax4 = fig[1, 2] = GLMakie.Axis(fig, title = "Orientation Filter Transform")
img4 = image!(ax4, Images.scaleminmax(0, maximum(jing_oft)).(jing_oft))
fig
##

##
fig = Figure(resolution = (1000, 700))
ax1 = fig[1, 1] = GLMakie.Axis(fig, title = "Actin (Equalized)")
img1 = image!(ax1, actin_adj[z_ind, :, :])
ax2 = fig[2, 1] = GLMakie.Axis(fig, title = "Actin (LFT intensity)")
img2 = image!(ax2, Images.scaleminmax(0, maximum(actin_I)).(actin_I))
ax3 = fig[2, 2] = GLMakie.Axis(fig, title = "Actin (LFT orientation)")
img3 = image!(ax3, angle_to_color.(actin_θ))
ax4 = fig[1, 2] = GLMakie.Axis(fig, title = "Actin (OFT)")
img4 = image!(ax4, Images.scaleminmax(0, maximum(actin_oft)).(actin_oft))
fig
##
fig = Figure(resolution = (1000, 700))
ax1 = fig[1, 1] = GLMakie.Axis(fig, title = "Desmin (Equalized)")
img1 = image!(ax1, desmin_adj[z_ind, :, :])
ax2 = fig[2, 1] = GLMakie.Axis(fig, title = "Desmin (LFT intensity)")
img2 = image!(ax2, Images.scaleminmax(0, maximum(desmin_I)).(desmin_I))
ax3 = fig[2, 2] = GLMakie.Axis(fig, title = "Desmin (LFT orientation)")
img3 = image!(ax3, angle_to_color.(desmin_θ))
ax4 = fig[1, 2] = GLMakie.Axis(fig, title = "Desmin (OFT)")
img4 = image!(ax4, Images.scaleminmax(0, maximum(desmin_oft)).(desmin_oft))
fig
##
fig = Figure(resolution = (800, 700))
ax1 = fig[1, 1] = GLMakie.Axis(fig, title = "Overlay (Equalized)")
img1 = image!(ax1, Images.colorview(Images.RGB, desmin_adj, actin_adj, Images.zeroarray)[z_ind, :, :])
ax2 = fig[2, 1] = GLMakie.Axis(fig, title = "Actin (LFT intensity)")
img2 = image!(ax2, Images.colorview(Images.RGB, Images.scaleminmax(0, maximum(desmin_I)).(desmin_I), Images.scaleminmax(0, maximum(actin_I)).(actin_I), Images.zeroarray))
fig
##
import ImageBinarization: binarize, Otsu
lft_bin = binarize(desmin_I, Otsu())
image(lft_bin)
##
size(actin_adj)
##
fig = Figure(resolution = (1500, 1000))
ax1 = fig[1, 1] = Axis3(fig, title = "Actin 3D stack")
ax2 = fig[1, 2] = Axis3(fig, title = "Desmin 3D stack")
volume!(ax1, actin_adj[1:100, 1:100, 1:100], algorithm = :mip, colormap = :Greens_9, transparancy = true)
volume!(ax2, desmin_adj[1:100, 1:100, 1:100], algorithm = :mip, colormap = :Reds_9, transparancy = true)
fig
##
actin3dI, actin3dθ = LineFilterTransform(actin_adj[1:100, 1:100, 1:100], 10, 5, 10)
##
volume(Images.scaleminmax(0.0, maximum(actin3dI)).(actin3dI), algorithm = :mip, colormap = :Greens_9, transparancy = true)
##
desmin3dI, desmin3dθ = LineFilterTransform(desmin_adj[1:100, 1:100, 1:100], 10, 5, 10)
##
volume(Images.scaleminmax(Images.otsu_threshold(desmin3dI), maximum(desmin3dI)).(desmin3dI), algorithm = :mip, colormap = :Reds_9, transparancy = true)
##
fig = Figure(resolution = (1000, 700))
ax1 = fig[1, 1] = Axis3(fig, title = "Actin LFT intensity")
ax2 = fig[1, 2] = Axis3(fig, title = "Desmin LFT intensity")
volume!(ax1, Images.scaleminmax(Images.otsu_threshold(actin3dI), maximum(actin3dI)).(actin3dI), algorithm = :mip, colormap = :Greens_9, transparancy = true)
volume!(ax2, Images.scaleminmax(Images.otsu_threshold(desmin3dI), maximum(desmin3dI)).(desmin3dI), algorithm = :mip, colormap = :Reds_9, transparancy = true)
fig
##
desmin3doft = OrientationFilterTransform(desmin3dI, desmin3dθ, 10, 5, 10)
##
actin3doft = OrientationFilterTransform(actin3dI, actin3dθ, 10, 5, 10)
##
##
fig = Figure(resolution = (1000, 700))
ax1 = fig[1, 1] = Axis3(fig, title = "Actin OFT intensity")
ax2 = fig[1, 2] = Axis3(fig, title = "Desmin OFT intensity")
volume!(ax1, Images.scaleminmax(0.0, maximum(actin3doft)).(actin3doft), algorithm = :mip, colormap = :Greens_9, transparancy = true)
volume!(ax2, Images.scaleminmax(0.0, maximum(desmin3doft)).(desmin3doft), algorithm = :mip, colormap = :Reds_9, transparancy = true)
fig
##
fig = Figure(resolution = (800, 700))
ax1 = fig[1, 1] = GLMakie.Axis(fig, title = "Overlay (Equalized)")
img1 = image!(ax1, Images.colorview(Images.RGB, desmin_adj, actin_adj, Images.zeroarray)[z_ind, :, :])
ax2 = fig[2, 1] = GLMakie.Axis(fig, title = "Actin (LFT intensity)")
img2 = image!(ax2, Images.colorview(Images.RGB, desmin3dI[z_ind, :, :], actin3dI[z_ind, :, :], Images.zeroarray))
fig