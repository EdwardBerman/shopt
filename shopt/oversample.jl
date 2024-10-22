using Images
function oversample_image(image, new_dim)
    # Create an empty array to store the oversampled image
    oversampled_image = zeros(new_dim, new_dim)

    scale_factor = 1/(new_dim / size(image)[1])
    
    for y in 1:new_dim
      for x in 1:new_dim
            src_x = (x - 0.5) * scale_factor + 0.5
            src_y = (y - 0.5) * scale_factor + 0.5

            x0 = max(1, floor(Int, src_x))
            x1 = min(size(image)[1], x0 + 1)
            y0 = max(1, floor(Int, src_y))
            y1 = min(size(image)[2], y0 + 1)

            dx = src_x - x0
            dy = src_y - y0

            oversampled_image[y, x] =
                (1 - dx) * (1 - dy) * image[y0, x0] +
                dx * (1 - dy) * image[y0, x1] +
                (1 - dx) * dy * image[y1, x0] +
                dx * dy * image[y1, x1]
        end
    end

    return oversampled_image
end

function undersample_image(image, new_dim) 
    undersampled_image = imresize(image, new_dim, new_dim, algorithm = :bilinear)
    return undersampled_image
end
