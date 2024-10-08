# Preparation

Part of the first exercise is to setup the *environment* for running an analysis with Julia and using git and github to share and store code.

Instructions are highlighted as follows

> [!NOTE] 
> To install Julia on your system, follow [these instructions](https://julialang.org/downloads/).
> For git, go to https://git-scm.com/downloads.

While tips look like this:

> [!TIP]
> In a terminal use the commands `which julia` or `which git` (linux, macOS) `where.exe ...` (Windows) to check if your system knows the software and where it is installed. Note that a restart after installation might be needed.

> [!TIP]
> While you will be going through the setup together in the first tutorial, you might still get stuck with the software or the exercises. If that happens, do not hesitate to contact `marian.stahl@cern.ch`.

After the installation, we are ready to start a *project* in Julia. In a terminal, navigate to the directory you would like to store your work, and open the Julia command line interface (called [REPL](https://docs.julialang.org/en/v1/manual/getting-started/)). To understand the syntax, read the hint that follows the instructions.

## Setup julia

> [!NOTE] 
> ```
> $ julia
> julia> # this is a comment. Type ] to get to the package mode
> (@v1.10) pkg> generate DataAnalysisWS2425
> julia> cd("DataAnalysisWS2425")
> (@v1.10) pkg> activate .
> (DataAnalysisWS2425) pkg> add Plots DataFrames Pluto Statistics LinearAlgebra QuadGK Parameters Test Random
> julia> mkdir("notebooks")
> julia> cd("notebooks")
> julia> import Pluto
> julia> Pluto.run()
> ```

> [!TIP]
> `$` signals that the command is executed in the terminal <br>
> `julia>` is the julia command line interface (REPL). If you are in a different *mode*, type <backspace> to get back to this main mode. <br>
> `(@v1.10) pkg>` or `(DataAnalysisWS2425) pkg>` is the package mode. You can enter the package mode upon typing `]`. See [the REPL documentation](https://docs.julialang.org/en/v1/stdlib/REPL/) for details. <br>
> `shell>` is the shell mode (type `;`). We did not use it in the instructions, but you can also `mkdir` and `cd` in shell mode, instead of calling the julia commands. <br>
> `help>` is the help mode (type `?`).

With the command `Pluto.run()`, a Pluto notebook opens in your default browser. 

> [!NOTE] 
> Click on `Save notebook ...` and call it `exercise1`. (Do this before entring any code!) <br>
> In the cell, enter
> ```
> begin
>     import Pkg
>     # activate the shared project environment
>     Pkg.activate(joinpath(@__DIR__, ".."))
>     # instantiate, i.e. make sure that all packages are downloaded
>     Pkg.instantiate()
>     # You do not need all packages that you have added to your project earlier
> 	  # You can however add more as we move along. Just re-execute this cell and they will be there.
>     using DataAnalysisWS2425, Random, Plots
> end
> ```
> and execute it with <Shift+Enter>.

You are now ready to start coding on your local device with Julia, but an important step is still missing...

## Setup git

As a researcher, it is important to collaborate and make your code publically available. To do that, we will use github as platform to store and share code; this includes handing in your exercises. 
    
> [!TIP]
> You can find tutorials on git [here](https://git-scm.com/docs/gittutorial), on github [here](https://docs.github.com/en), or by searching for `git tutorial` with your favourite search engine.


> [!NOTE] 
> Exit Pluto by pressing <Ctrl+C>, then exit julia by pressing <Ctrl+D>.
>     
> Go to github.com and create an account if you don't already have one. <br>
> Create a new repository called `DataAnalysisWS2425`.
>     
> In the terminal, navigate to the `DataAnalysisWS2425` directory that Julia created.<br>
> Call `git init`.<br>
> If you get a hint message from git, you can follow it:
> ```
> git config --global init.defaultBranch main
> git branch -m main
> ```
> While at it, configure user mail and name:
> ```
> git config --global user.email "you@example.com"
> git config --global user.name "Your Name"
> ```
> You can now add the project to the new repository with `git add *`.<br>
> If you type `git status`, you should see
> ```
> On branch main
> 
> No commits yet
> 
> Changes to be committed:
>   (use "git rm --cached <file>..." to unstage)
>         new file:   Manifest.toml
>         new file:   Project.toml
>         new file:   notebooks/exercise1.jl
>         new file:   src/DataAnalysisWS2425.jl
> ```
> You can do the first *commit* now (adding a descriptive message with `-m`)
> ```
> git commit -m 'setup for data analysis exercises'
> ```
>  
> Before you can release the code in the open, you need to be able to authentificate yourself in github. <br>
> To do that, follow the steps to <br>
> [generate a ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key), <br>
> [add the key to the local ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent), and <br>
> [add your key to github](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account#adding-a-new-ssh-key-to-your-account).
> 
> It's time to make the code public:
> ```
> git remote add origin git@github.com:<your_github_user_name>/DataAnalysisWS2425.git
> git push -u origin main
> ```

> [!TIP]
> Commits are snapshots of your code with a unique identifier (hash). You can always go back to these snapshots that you have created.
>     
> Adding your ssh key to the local ssh agent needs to be repeated every time you restart your computer, so it makes sense to add these lines to `~/.bashrc` or equivalents. You can also create a shell function in your `bashrc` to setup the environment for these exercises.

## Back to julia, running tests

To get back working on your notebook, open julia in the main directory (`DataAnalysisWS2425`) of your project, revive the project, and start Pluto:
```
(@v1.10) pkg> activate .
julia> cd("notebooks")
julia> import Pluto; Pluto.run()
```

In the upcoming tutorial sessions, you will be implementing *tests* to check if the your code works as expected.

Here is how you can run a dummy test:
> [!NOTE]
> Create the file `test/runtests.jl` in your main directory (`DataAnalysisWS2425`) with content
> ```
> using DataAnalysisWS2425
> using Test
> @testset "Fake test" begin
>    @test 1 == 1
> end
> ```
> Run tests for the project from a julia REPL (assuming your current directory is `DataAnalysisWS2425`)
> ```
> (@v1.10) pkg> activate .
> (DataAnalysisWS2425) pkg> test
> ```
> In the end, you should see
> ```
>      Testing Running tests...
> Test Summary: | Pass  Total  Time
> Fake test     |    1      1  0.0s
>     Testing DataAnalysisWS2425 tests passed
> ```
