module ZStackAnalysis

include("./Transforms.jl")
using Interpolations
using .Transforms

function interpolate_zstack(zstack, scale)
    scale_x, scale_y, scale_z = scale/minimum(scale)
    nz, nx, ny = size(zstack)
    grids = ((0:nz-1)*scale_z, (0:nx-1)*scale_x, (0:ny-1)*scale_y)
    algorithm = BSpline(Linear())
    itp = interpolate(zstack, algorithm)
    sitp = Interpolations.scale(itp, grids...)
    return sitp(0:(nz-1)*scale_z, 0:(nx-1)*scale_x, 0:(ny-1)*scale_y)
end

export interpolate_zstack
export LineFilterTransform, OrientationFilterTransform
end
