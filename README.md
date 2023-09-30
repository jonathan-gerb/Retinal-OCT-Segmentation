# Fundus-OCT-challenge
Repository for the IChallenge Goals dataset: Fundus OCT layer segmentation

# install guide
make a new conda environment:
```
conda create -n fundus-oct python=3.10
```

install the requirements for the repository:
```
pip install -r requirements.txt
```

now install the codebase itself, I suggest installing it locally for development
```
pip install -e .
```

# usage guide
After installing the package can be called using the following shorthand:
```
fundus-oct --cfg /path/to/config/file.yml
```
Experiments are defined using config yaml files that overwrite the default arguments. To define your own experiment create a new config file `new_config.yml`. For an example see the `basic_config.yml` in `run/configs`. 
A full list of arguments is defined in `fundus_oct_challenge/config/defaults.py`.

### Weights and Biases
Logging is done using weights and biases. To use weights and biases, make and account and login to it from the command line. see https://docs.wandb.ai/quickstart 
Additionally you'll have to change the `wandb.init()` call in the `__main__.py` file. We'll change how that works soon though.


