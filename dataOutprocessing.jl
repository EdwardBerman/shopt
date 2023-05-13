## .shopt file
function writeData(size, shear1, shear2, sizeD, shear1D, shear2D)
  df = DataFrame(star = 1:length(size), 
                 s_model=size, 
                 g1_model=shear1, 
                 g2_model=shear2, 
                 s_data=sizeD, 
                 g1_data=shear1D, 
                 g2_data=shear2D)

  CSV.write(joinpath("outdir", "df.shopt"), df)
end


function readData()
  DataFrame(CSV.File(joinpath("outdir", "df.shopt")))
end

