# Relaxed Adaptive Projection
Hello! This GitHub repository contains the source code for the paper [Private synthetic data for multitask learning and marginal queries](https://arxiv.org/pdf/2209.07400.pdf).

Our paper ran experiments on the American Community Survey datasets using the same pre-processing as the
[Vietri et al. 20](http://proceedings.mlr.press/v119/vietri20b/vietri20b.pdf) and [McKenna et al. 2019](https://arxiv.org/abs/1901.09136) papers.

## Requirements and Setup
Our project can be run on CPU and GPU, and the necessary python packages can be installed through
`pip install -r requirements.txt`


## Datasets
Datasets can be downloaded from [folktables](https://github.com/socialfoundations/folktables). Our code automatically downloads survey data from 2014. To download the data from a different year, simply passing the argument `survey_year=2018` to the data loading function `get_acs`.

## Running the data generator
`main.py` is the entrypoint for running experiments/generating data.

An example invocation to run an experiment on ACS dataset and specifically using the data for income data in California:
`python main.py --states CA --tasks income`

An example invocation to generate differentially privacy synthetic data on the data for multiple tasks in California:
`python main.py --states CA --multitask`

The default algorithm is RAP++, and we also support RAP in our implementation, see the example below:
`python main.py --states CA --tasks income --algorithm RAP`

To access the script usage, run: `python main.py -h`


## Security

See CONTRIBUTING for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Citation
Please use the following citation when publishing material that uses our code:
```tex
@inproceedings{
vietri2022private,
title={Private Synthetic Data for Multitask Learning and Marginal Queries},
author={Giuseppe Vietri and Cedric Archambeau and Sergul Aydore and William Brown and Michael Kearns and Aaron Roth and Ankit Siva and Shuai Tang and Steven Wu},
booktitle={Advances in Neural Information Processing Systems},
year={2022},
url={https://openreview.net/forum?id=5JdyRvTrK0q}
}
```

```tex
@InProceedings{pmlr-v139-aydore21a,
  title = 	 {Differentially Private Query Release Through Adaptive Projection},
  author =       {Aydore, Sergul and Brown, William and Kearns, Michael and Kenthapadi, Krishnaram and Melis, Luca and Roth, Aaron and Siva, Ankit A},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  year = 	 {2021},
  series = 	 {Proceedings of Machine Learning Research},
  pdf = 	 {http://proceedings.mlr.press/v139/aydore21a/aydore21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/aydore21a.html},
}
```
