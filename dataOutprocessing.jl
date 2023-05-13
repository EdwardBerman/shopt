## .shopt file
function writeData(size, shear1, shear2)
  df = DataFrame(star = 1:length(size), s=size, g1=shear1, g2=shear2)
  CSV.write(joinpath("outdir", "df.shopt"), df)
end


function readData()
  DataFrame(CSV.File(joinpath("outdir", "df.shopt")))
end

