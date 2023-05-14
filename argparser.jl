function process_arguments(args)
  fancyPrint("Parsing Arguments")
  configdir = args[1]
  println("\n\tConfig Directory: ", configdir)
  outdir = args[2]
  println("\tOutput Directory: ", outdir)
  datadir = args[3]
  println("\tData Directory: ", datadir,"\n")
end

