juliaup add 1.11.3
juliaup default 1.11.3
julia --project --eval '
    import Pkg
    Pkg.add("IJulia")
    import IJulia
    IJulia.installkernel("julia")
'
