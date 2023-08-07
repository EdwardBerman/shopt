function pca_image(image_path, ncomponents)
  # Load the image
  img_matrix = image_path

  # Perform PCA
  M = fit(PCA, img_matrix; maxoutdim=ncomponents)

  # Transform the image into the PCA space
  transformed = MultivariateStats.transform(M, img_matrix)

  # Reconstruct the image
  reconstructed = reconstruct(M, transformed)

  # Reshape the image back to its original shape
  reconstructed_image = reshape(reconstructed, size(img_matrix)...)
end

