using Pluto
using Pkg

files = readdir("lectures")
# only lecture-
filter!(x -> occursin(r"lecture-", x), files)

map(files) do file
    Pluto.activate_notebook_environment(joinpath("lectures", file))
    Pkg.update()
end

# using PlutoSliderServer
# PlutoSliderServer.run_directory(joinpath("lectures"), SliderServer_enabled = false)
