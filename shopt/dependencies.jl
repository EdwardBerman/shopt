using Pkg

function is_installed(package::AbstractString)
    try
        Pkg.installed(package)
        return true
    catch
        return false
    end
end

function download_packages(file_path::AbstractString)
    # Read the file and retrieve the import statements
    file = open(file_path, "r")
    imports = readlines(file)
    close(file)

    # Download and import the packages
    for package in imports
        package = strip(package)
        if !is_installed(package)
            try
                Pkg.add(package)
                println("Successfully downloaded and added $package")
            catch err
                println("Failed to download and add $package: $err")
                continue
            end
        end
        
        try
            eval(Meta.parse("import $package"))
            println("Successfully imported $package")
        catch err
            println("Failed to import $package: $err")
        end
    end
end

# Provide the file path of the imports.txt file
download_packages("imports.txt")

