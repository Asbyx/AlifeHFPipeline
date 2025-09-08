# Alife RLHF pipeline

> [!WARNING]
> **This branch is made for the late breaking abstracts for the Alife 2025 Conference !**
> 
> This repository contains experimental code for research purposes. The codebase is actively under development and may contain bugs, incomplete features or bad documentation.
> 
> - **Not production-ready**: This software is intended for research and experimentation only.
> - **Limited support**: This is a research project with limited maintenance resources.
> - **Contributions welcome**: Feel free to open issues or submit pull requests to help improve the project !
>
> This branch will be terminated when the project will be publication ready.


## Presentation
The idea of this app is to be a pipeline applying RLHF to train a model for generating interesting simulation parameters.

The pipeline is a loop of 4 stages:
- **Generation**: A Generator produces some parameters
- **Labeling**: A human makes a ranking of the output of the simulations
- **Training**: A Rewarder is trained to approximate the ranking of a human
- **Fine Tuning**: Using the Rewarder, train the Generator.
- and repeat...
This pipeline include every steps.  

This pipeline is fully in python.

## Installation
Clone the repository.  
Run `pip install -e .` in order to install the required packages and the `rlhfalife` package which you can easily import in any files to use. It is not expected to import anything else than the `rlhfalife.utils` module.  
Happy coding !

## Usage
One must implement a **python package**, containing a `Simulator`, a `Generator` and a `Rewarder`, in `profiles/<new profile>`, using the abstract classes given in `rlhfalife.utils.py` (it is recommended to inherit the classes).  
The parameters of the functions to implement **must not be changed**, as the pipeline make automated calls to them. But, one can define the `__init__` as they want, as long as the super is also called. That way, one can add the attributes they want to their object. The initialization of the objects is fully handled by the user through the `Loader`. One can also create functions called by other components.   

**The only 2 requirements of the package** are: 
- Have a `Loader` (abstract still available in `rlhfalife.utils.py`, and should be named `Loader` as well), which is made available through an `__init__.py` file in the package.  
- Have a `configs` folder containing `json` config files, that will be passed to your `Loader`, with which you can setup your `Simulator`, a `Generator` and a `Rewarder` as you wish.     
The rest is free to implement as your convenience, using the file architecture you want !

An example is provided with the `lenia` profile. Note that the files structure is not mandatory, only the `__init__.py` should be present.
Then, by running the main, the full pipeline is launched and the training starts !  
*N.B: uppon running `python main.py`, you will be prompted to select a profile and a config. But you can pass them by arguments: `python main.py --profile lenia --config default`* 


The pipeline is divided in three categories:
- **Label videos**: A window with 2 simulations will be presented, the user must select the "most interesting" one. Videos generation is fully automated.
- **Benchmark rewarder**: 10 simulations will be generated and presented to the user, along with their scores given by the rewarder. That way, the user can check that the rewarder is indeed well aligned with the expectations.
- **Launch training**: Simply launch a rewardor and then a generator training, based on the labeled simulations.

Everything is saved in `out/<profile>/<config>/`. That way you can recover parameters, outputs and videos easily.   

Please, fill free to open an issue to report a bug or make a suggestion !  
