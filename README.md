# Relaxed Adaptive Projection
Hello! This GitHub repository contains the source code for the paper [Differentially Private Query Release Through Adaptive Projection](https://arxiv.org/abs/2103.06641).

Our paper ran experiments on the ADULT and LOANS datasets using the same pre-processing as the 
[Vietri et al. 20](http://proceedings.mlr.press/v119/vietri20b/vietri20b.pdf) and [McKenna et al. 2019](https://arxiv.org/abs/1901.09136) papers.

## Requirements and Setup
Our project can be run on CPU and GPU. We have set up Dockerfiles for both cases but feel free to use Conda/venv/the package manager of your choice.

### Docker CPU
Build the docker image by running (substituting `<image_name>` with your choice of name):
```bash
docker build -t <image_name> .
```

Then you can start a shell in the container with the source directory volume mapped to `/usr/src` by:
```bash
docker run --rm -itv $(pwd):/usr/src <image_name> /bin/bash
```

If you wish to instead start a Python REPL in the container:
```bash
docker run --rm -itv $(pwd):/usr/src <image_name> /bin/bash
```

### Docker GPU
This option requires that the NVidia Docker runtime be installed. This is standard in most Deep Learning based VMs (eg: DLAMI on AWS).
Build the GPU docker image by running (substituting `<image_name>` with your choice of name):
```bash
docker build -t <image_name> -f Dockerfile.gpu .
```

Then you can start a shell in the container with the source directory volume mapped to `/usr/src` by:
```bash
nvidia-docker run --rm -itv $(pwd):/usr/src <image_name> /bin/bash
```

If you wish to instead start a Python REPL in the container:
```bash
nvidia-docker run --rm -itv $(pwd):/usr/src <image_name> /bin/bash
```

### Local CPU
To install the CPU version of our code locally, clone this repository and then run:
```bash
pip install -r requirements.txt
```

### Local GPU
In order to install the GPU version of our code locally, you will need to install all requirements but `jaxlib`. Run:
```bash
grep -v "jax" requirements.txt | xargs pip install
```

And then, find the version of CUDA that's installed on your machine by running
```bash
nvcc --version
```

Finally, follow the instructions at the [JAX Installation Guide](https://github.com/google/jax#pip-installation-gpu-cuda).

## Datasets
Download the dataset `csv`s and corresponding `-domain.json` files from the following links and place datasets in the (empty) `data` folder.
1. [ADULT](https://github.com/ryan112358/private-pgm/tree/master/data)
1. [LOANS](https://github.com/giusevtr/fem/tree/master/datasets)

## Running the data generator
`main.py` is the entrypoint for running experiments/generating data.

An example invocation to run an experiment on adult dataset, to generate 800 points with learning rate of 0.1:
`python main_iterative.py --data-source adult --num-generated-points 1000 --epochs 5 --top-q 5 --seed 0 --statistic-module statistickway --k 3 --workload 64 --learning-rate 1e-3`

You are also free to use config files like:
`python main_iterative.py -c adult_config.txt`


To access the script usage listed below, run: `python main.py -h`

## Usage
```
usage: main.py [-h] [--config-file CONFIG_FILE] [--num-dimensions D] [--num-points N] [--num-generated-points N_PRIME] [--epsilon EPSILON] [--delta DELTA] [--iterations ITERATIONS] [--save-figures SAVE_FIG]
               [--no-show-figures NO_SHOW_FIG] [--ignore-diagonals IGNORE_DIAG] [--data-source {toy_binary,adult,loans}] [--read-file READ_FILE] [--use-data-subset USE_SUBSET] [--filepath FILEPATH]
               [--destination_path DESTINATION] [--seed SEED] [--statistic-module STATISTIC_MODULE] [--k K] [--workload WORKLOAD] [--learning-rate LEARNING_RATE] [--project [PROJECT [PROJECT ...]]]
               [--initialize_binomial INITIALIZE_BINOMIAL] [--lambda-l1 LAMBDA_L1] [--stopping-condition STOPPING_CONDITION] [--all-queries] [--top-q TOP_Q] [--epochs EPOCHS] [--csv-path CSV_PATH] [--silent]
               [--verbose] [--norm {Linfty,L2,L5,LogExp}] [--categorical-consistency] [--measure-gen] [--oversamples OVERSAMPLES]

Args that start with '--' (eg. --num-dimensions) can also be set in a config file (specified via --config-file). Config file syntax allows: key=value, flag=true, stuff=[a,b,c] (for details, see syntax at
https://goo.gl/R74nmi). If an arg is specified in more than one place, then commandline values override config file values which override defaults.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE, -c CONFIG_FILE
                        Path to config file
  --num-dimensions D, -d D
                        Number of dimensions in the original dataset. Does not need to be set when consuming csv files (default: 2)
  --num-points N, -n N  Number of points in the original dataset. Only used when generating datasets (default: 1000)
  --num-generated-points N_PRIME, -N N_PRIME
                        Number of points to generate (default: 1000)
  --epsilon EPSILON     Privacy parameter (default: 1)
  --delta DELTA         Privacy parameter (default: 1/n**2)
  --iterations ITERATIONS
                        Number of iterations (default: 1000)
  --save-figures SAVE_FIG
                        Save generated figures
  --no-show-figures NO_SHOW_FIG
                        Not show generated figuresduring execution
  --ignore-diagonals IGNORE_DIAG
                        Ignore diagonals
  --data-source {toy_binary,adult,loans}
                        Data source used to train data generator
  --read-file READ_FILE
                        Choose whether to regenerate or read data from file for randomly generated datasets
  --use-data-subset USE_SUBSET
                        Use only n rows and d columns of the data read from the file as input to the algorithm. Will not affect random inputs.
  --filepath FILEPATH   File to read from
  --destination_path DESTINATION
                        Location to save figures and configuration
  --seed SEED           Seed to use for random number generation
  --statistic-module STATISTIC_MODULE
                        Module containing preserve_statistic function that defines statistic to be preserved. Function MUST be named preserve_statistic
  --k K                 k-th marginal (default k=3)
  --workload WORKLOAD   workload of marginals (default 64)
  --learning-rate LEARNING_RATE, -lr LEARNING_RATE
                        Adam learning rate (default: 1e-3)
  --project [PROJECT [PROJECT ...]]
                        Project into [a,b] b>a during gradient descent (default: None, do not project))
  --initialize_binomial INITIALIZE_BINOMIAL
                        Initialize with 1-way marginals
  --lambda-l1 LAMBDA_L1
                        L1 regularization term (default: 0)
  --stopping-condition STOPPING_CONDITION
                        If improvement on loss function is less than stopping condition, RAP will be terminated
  --all-queries         Choose all q queries, no selection step. WARNING: this option overrides the top-q argument
  --top-q TOP_Q         Top q queries to select (default q=500)
  --epochs EPOCHS       Number of epochs (default: 100)
  --csv-path CSV_PATH   Location to save results in csv format
  --silent, -s          Run silently
  --verbose, -v         Run verbose
  --norm {Linfty,L2,L5,LogExp}
                        Norm to minimize if using the optimization paradigm (default: L2)
  --categorical-consistency
                        Enforce consistency categorical variables
  --measure-gen         Measure Generalization properties
  --oversamples OVERSAMPLES
                        comma separated values of oversamling rates (default None)
```

## Security

See CONTRIBUTING for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Citation
Please use the following citation when publishing material that uses our code:
```tex
@misc{aydore2021differentially,
      title={Differentially Private Query Release Through Adaptive Projection}, 
      author={Sergul Aydore and William Brown and Michael Kearns and Krishnaram Kenthapadi and Luca Melis and Aaron Roth and Ankit Siva},
      year={2021},
      eprint={2103.06641},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
