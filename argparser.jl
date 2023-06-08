function process_arguments(args)
  fancyPrint("Parsing Arguments")
  configdir = args[1]
  println("\n\tConfig Directory: ", configdir)
  outdir = args[2]
  println("\tOutput Directory: ", outdir)
  catalog = args[3]
  println("\tCatalog: ", catalog,"\n")
  sci_file = args[4]
  println("\tScience File: ", sci_file,"\n")
end

