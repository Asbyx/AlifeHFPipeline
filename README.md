# Alife HF pipeline
## Presentation
The idea of this app is to be a pipeline applying Reward Modeling using Human Preferences to train a model for generating interesting simulation parameters.

The pipeline is a loop of 4 stages:
- **Generation**: A Generator produces some parameters  
- **Labeling**: A human makes a ranking of the output of the simulations that are run with those parameters  
- **Training**: A Rewarder is trained to approximate the ranking of a human  
- **Fine Tuning**: Using the Rewarder, train the Generator.  
- and repeat...  

This pipeline includes every steps.   

This pipeline is fully in python.

## Installation
Clone the repository.  
Run `pip install -r requirements.txt` and `pip install -e .` in order to install the required packages and the `rlhfalife` package which you can easily import in any files to use. It is not expected to import anything else than the `rlhfalife.utils` module.  
Happy coding !

## Usage
One must implement a **python package**, containing a `Loader`, a `Simulator`, a `Generator` and a `Rewarder`, in `profiles/<new profile>`, using the abstract classes given in `rlhfalife.utils.py` (it is recommended to inherit the classes).  
The parameters of the functions to implement **must not be changed**, as the pipeline make automated calls to them. But, one can define the constructor (`__init__`) as they want, as long as the super is also called (except for the `Loader`, that cannot take any parameter). That way, one can add the attributes they want to their object.  
The initialization of the objects is fully handled by the user through the `Loader`. One can also create functions called by other components.   

**The only 2 requirements of the package** are: 
- Have a `Loader` (abstract also available in `rlhfalife.utils.py`, and should be named `Loader` as well), which is made available through an `__init__.py` file in the package.  
- Have a `configs` folder containing `json` config files, that will be passed to your `Loader`, with which you can setup your `Simulator`, a `Generator` and a `Rewarder` as you wish.     
The rest is free to implement as your convenience, using the file architecture you want !

An example is provided with the `lenia` profile. Note that the files structure is not mandatory, only the `__init__.py` should be present.  
Then, by running the main, the full pipeline is launched and the training starts !  
*N.B: upon running `python main.py`, you will be prompted to select a profile and a config. But you can pass them by arguments: `python main.py --profile lenia --config default`* 


## Breakdown of the main menu:
### 1. Data Labeling
- **11. Label Pairs (needs GUI)**: Label videos by pairs.
- **12. Quad Labeler (needs GUI)**: Label videos by doing rankings of 4 videos

### 2. Training & Generation
- **21. Launch training**: Train the Rewarder on the labeled dataset.
- **22. Online training (needs GUI)**: Interactive training loop where labeling and training phases are alterning.
- **23. Generate pairs (no GUI)**: Use the current Generator to produce new simulation parameters and simulate them, storing the videos for future labeling.
- **24. Generate videos with current generator (no GUI)**: Generate standalone videos to inspect the current state of your Generator.
- **25. Benchmark rewarder**: Evaluate the performance of your Rewarder model, create benchmarks or test the current Rewarder on them.

### 3. Profile & Configuration
- **31. Export profile**: Export your current profile models and data into a zip file.
- **32. Reload new code**: Hot-reload your profile code (e.g., changes to Simulator/Generator/Rewarder) without restarting the CLI.
- **33. Reload models and data managers**: Hot-reload the configuration.
- **34. Change frame size**: Adjust the displayed frame size for the videos.
- **35. Custom script**: Execute any custom scripts defined in the loader.

### 4. Data Management
- **41. Analyze training dataset**: Print statistics and analyze the composition of your labeled preferences dataset.
- **42. Reset labels**: Clear the labels.
- **43. Reset config**: Completely erase all data for the current configuration profile, including models, parameters, videos, and labels.

## ./examples

You can view a gallery of generated videos comparing the results from random priors versus parameters generated using Human Preference Reward Models on our dedicated examples page:

**[View the Interactive Examples Gallery](https://Asbyx.github.io/AlifeHFPipeline/)**  
