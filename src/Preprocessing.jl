module Preprocessing
using Base.Threads
using Images

function convert_to_float32(zstack::AbstractArray)
    return Float32.(reinterpret(Images.N0f16, zstack))
end

function equalize_zstack(zstack)
    # equalize zstack with equalization separately performed for each z slice
    output = similar(zstack)
    equalizer = Images.Equalization(nbins = 256, minval = 0.0, maxval = 0.7)
    @threads for I in CartesianIndices(axes(zstack)[1:2])
        output[I, :, :] = Images.adjust_histogram(zstack[I, :, :], equalizer)
    end
    return output
end

export convert_to_float32
end