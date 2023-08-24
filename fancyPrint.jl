#=
Making Log Statements Look Nicer, fancyPrints signify stepping stones in the program
=#

function fancyPrint(printStatement)
  for i in 1:(length(printStatement) + 4)
    print("-") 
  end
    println("\n| "*printStatement*" |")
  for i in 1:(length(printStatement) + 3)
    print("-") 
  end
  println("-")
end

function fancyPrintTwo(printStatement)
  for i in 1:(length(printStatement) + 4)
    print("#") 
  end
    println("\n# "*printStatement*" #")
  for i in 1:(length(printStatement) + 3)
    print("#") 
  end
  println("#")
end

