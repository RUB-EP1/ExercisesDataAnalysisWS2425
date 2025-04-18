# DataAnalysisWS2425

Welcome to the exercises for the "Data Analysis in HEP" lecture! :wave:

In these exercises you will work through the statistical aspects of a typical data analysis in high energy physics. You will be equipped with a unique data set that resembles real data taken with the [LHCb experiment](https://lhcb-outreach.web.cern.ch/).

During the course, you will develop a code base for this analysis in [Julia](https://julialang.org/) starting from scratch.

Before getting started with the exercises, you will need to follow the [setup](exercises/setup.md) steps to prepare the software environment on your machine.

## Lecture Material

Code for the notebooks showcased at the lectures can be found in the [lectures/](lectures/) folder, see [lectures/README.md](lectures/README.md) for overview.
The github link can be used for the path link in Pluto starting page.

## Exercises and Tutorials

The exercise sheets are distributed on Wednesday of the tutorial, and due on Sunday 1.5 weeks later.
Solution of the problems and the code should be sent to <Dhruvanshu.Parmar@ruhr-uni-bochum.de> (link to a code, pdf of a notebook).

- [Sheet-01](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-01.md), published on **9/10/2024**, due on **20/10/2024**, discussed at the [Tutorial-02](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/tutorials/tutorial-02.md)
- [Sheet-02](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-02.md), published on **25/10/2024**, due on **3/11/2024**, discussed at the [Tutorial-03](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/tutorials/tutorial-03.md)
- [Sheet-03](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-03.md), published on **11/11/2024**, due on **17/11/2024**, discussed at the [Tutorial-04](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/tutorials/tutorial-04.md)
- [Sheet-04](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-04.md), published on **26/11/2024**, due on **08/12/2024**, discussed at the [Tutorial-05](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/tutorials/tutorial-05.md)
- [Sheet-05](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-05.md), published on **10/12/2024**, due on **17/12/2024**, discussed at the [Tutorial-06](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/tutorials/tutorial-06.md)
- [Sheet-06](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-06.md), published on **17/12/2024**, due on **19/01/2025**
- [Sheet-07](https://github.com/RUB-EP1/ExercisesDataAnalysisWS2425/blob/main/exercises/sheet-07.md): additional, handing it is voluntary.

## Reference DataAnalysisWS2425

> [!IMPORTANT]
> The julia package that inherits functionality developed during this course is separated into a standalone package.
> Use [`HighEnergyTools.jl`](https://github.com/RUB-EP1/HighEnergyTools.jl) for basic sampling, fitting and plotting.

The project is a valid julia Julia module, that is being developed along with the course.
It provides essential tools and examples for statistical data analysis and fitting, designed to support and complement the lecture materials.
Many lecture notebooks start with a dependency cell that includes,

```julia
let
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    Pkg.instantiate()
end
```

referring to the functionality implemented in the module. [Read more about environments.](https://plutojl.org/en/docs/packages-advanced/)

Students' repository have a similar structure, with the exercises and tutorials focused on the homework.
Participants of the course are encouraged to review and build upon the implementation here to strengthen their grasp of Data Analysis.

## Technical notes

### Pre-commit

This repository uses [pre-commit](https://pre-commit.com/) to check the code and text.
To run the checks locally,

```bash
pre-commit run -a
```

### Git: delete branches that are gone

Once the PR is merged, and branch is delete on remote, it is a good idea to clear up the local copy.
It is done by pruning branches, following deleting these that are gone.

```bash
git fetch --prune
git branch -vv | grep ': gone]' | awk '{print $1}' | xargs git branch -D
```

## License

This project is licensed under the MIT License.
