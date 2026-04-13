# Alife RLHF pipeline
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

Here is a comparison of videos generated with a random prior parameter generator versus a trained reward model.

### Lenia

| Prior | Reward Model |
|:---:|:---:|
| ![Video Prior 1](./examples/lenia/using_prior/129029842981.mp4) | ![Video RM 1](./examples/lenia/using_reward_models/108833336161.mp4) |
| ![Video Prior 2](./examples/lenia/using_prior/129060200640.mp4) | ![Video RM 2](./examples/lenia/using_reward_models/108833336261.mp4) |
| ![Video Prior 3](./examples/lenia/using_prior/129060201150.mp4) | ![Video RM 3](./examples/lenia/using_reward_models/108833336281.mp4) |
| ![Video Prior 4](./examples/lenia/using_prior/129060203185.mp4) | ![Video RM 4](./examples/lenia/using_reward_models/108833364443.mp4) |
| ![Video Prior 5](./examples/lenia/using_prior/129060220805.mp4) | ![Video RM 5](./examples/lenia/using_reward_models/127329389843.mp4) |

### Particle Life

| Prior | Reward Model |
|:---:|:---:|
| ![Video Prior 1](./examples/particlelife/using_prior/-106138492953067878.mp4) | ![Video RM 1](./examples/particlelife/using_reward_models/-2728612294673905733.mp4) |
| ![Video Prior 2](./examples/particlelife/using_prior/-2369147273362576652.mp4) | ![Video RM 2](./examples/particlelife/using_reward_models/-8803982393252838527.mp4) |
| ![Video Prior 3](./examples/particlelife/using_prior/-4499023261873601467.mp4) | ![Video RM 3](./examples/particlelife/using_reward_models/-881289731860236246.mp4) |
| ![Video Prior 4](./examples/particlelife/using_prior/-6703622387425993567.mp4) | ![Video RM 4](./examples/particlelife/using_reward_models/1000960850808303495.mp4) |
| ![Video Prior 5](./examples/particlelife/using_prior/1751419491931583468.mp4) | ![Video RM 5](./examples/particlelife/using_reward_models/1006960128830633922.mp4) |

### Macelenia  
Using a trained model:    
![Video RM 1](./examples/macelenia/using_reward_models/141342322269.mp4)  
![Video RM 2](./examples/macelenia/using_reward_models/141342322425.mp4)  
![Video RM 3](./examples/macelenia/using_reward_models/141342322449.mp4)  
![Video RM 4](./examples/macelenia/using_reward_models/141342322497.mp4)  
