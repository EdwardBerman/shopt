#=
Function to parse arguments from the command line
=#

function process_arguments(args)
  fancyPrint("Parsing Arguments")
  configdir = args[1]
  println("━ Config Directory: ", configdir)
  outdir = args[2]
  println("━ Output Directory: ", outdir)
  catalog = args[3]
  println("━ Catalog: ", catalog)
end

