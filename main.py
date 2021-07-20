# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
import logging
import os

import configargparse
import pandas as pd
from jax import numpy as np, random

from relaxed_adaptive_projection import RAPConfiguration, RAP
from relaxed_adaptive_projection.constants import Norm, ProjectionInterval
from utils_data import data_sources, ohe_to_categorical

parser = configargparse.ArgumentParser()
parser.add_argument(
    "--config-file",
    "-c",
    required=False,
    is_config_file=True,
    help="Path to config file",
)
parser.add_argument(
    "--num-dimensions",
    "-d",
    type=int,
    default=2,
    dest="d",
    help="Number of dimensions in the "
    "original dataset. Does not need to "
    "be set when consuming csv files ("
    "default: 2)",
)

parser.add_argument(
    "--num-points",
    "-n",
    type=int,
    default=1000,
    dest="n",
    help="Number of points in the original "
    "dataset. Only used when generating "
    "datasets (default: 1000)",
)

parser.add_argument(
    "--num-generated-points",
    "-N",
    type=int,
    default=1000,
    dest="n_prime",
    help="Number of points to " "generate (default: " "1000)",
)

parser.add_argument(
    "--epsilon", type=float, default=1, help="Privacy parameter (default: 1)"
)
parser.add_argument(
    "--delta", type=float, default=None, help="Privacy parameter (default: 1/n**2)"
)
parser.add_argument(
    "--iterations", type=int, default=1000, help="Number of iterations (default: 1000)"
)

parser.add_argument(
    "--save-figures",
    type=bool,
    default=False,
    dest="save_fig",
    help="Save generated figures",
)
parser.add_argument(
    "--no-show-figures",
    type=bool,
    default=False,
    dest="no_show_fig",
    help="Not show generated figures" "during execution",
)

parser.add_argument(
    "--ignore-diagonals",
    type=bool,
    default=False,
    dest="ignore_diag",
    help="Ignore diagonals",
)
parser.add_argument(
    "--data-source",
    type=str,
    choices=data_sources.keys(),
    default="toy_binary",
    dest="data_source",
    help="Data source used to train data generator",
)
parser.add_argument(
    "--read-file",
    type=bool,
    default=False,
    help="Choose whether to regenerate or read data from file "
    "for randomly generated datasets",
)

parser.add_argument(
    "--use-data-subset",
    type=bool,
    default=False,
    dest="use_subset",
    help="Use only n rows and d "
    "columns of the data read "
    "from the file as input to "
    "the algorithm. Will not "
    "affect random inputs.",
)

parser.add_argument("--filepath", type=str, default="", help="File to read from")
parser.add_argument(
    "--destination_path",
    type=str,
    default="figures/",
    dest="destination",
    help="Location to save " "figures and " "configuration",
)
parser.add_argument(
    "--seed", type=int, default=0, help="Seed to use for random number generation"
)
parser.add_argument(
    "--statistic-module",
    type=str,
    default="statistickway",
    help="Module containing preserve_statistic "
    "function that defines statistic to be "
    "preserved. Function MUST be named "
    "preserve_statistic",
)
parser.add_argument("--k", type=int, default=3, help="k-th marginal (default k=3)")
# parser.add_argument('--num-rand-queries', type=int, default=None,
#                     help='number of random k-th queries (default None, i.e, enumerate all queries from marginals)')
parser.add_argument(
    "--workload", type=int, default=64, help="workload of marginals (default 64)"
)
parser.add_argument(
    "--learning-rate",
    "-lr",
    type=float,
    default=1e-3,
    help="Adam learning rate (default: 1e-3)",
)
parser.add_argument(
    "--project",
    nargs="*",
    type=float,
    default=None,
    help="Project into [a,b] b>a during gradient descent (default: None, do not project))",
)
parser.add_argument(
    "--initialize_binomial",
    type=bool,
    default=False,
    help="Initialize with 1-way marginals",
)
parser.add_argument(
    "--lambda-l1", type=float, default=0, help="L1 regularization term (default: 0)"
)
parser.add_argument(
    "--stopping-condition",
    type=float,
    default=10 ** -7,
    help="If improvement on loss function is less than stopping condition, RAP will be terminated",
)
parser.add_argument(
    "--all-queries",
    action="store_true",
    help="Choose all q queries, no selection step. WARNING: this option overrides the top-q argument",
)
parser.add_argument(
    "--top-q", type=int, default=50, help="Top q queries to select (default q=500)"
)
parser.add_argument(
    "--epochs", type=int, default=10, help="Number of epochs (default: 100)"
)
# parser.add_argument('--select-T', type=float, default=0.1,
#                     help='Threshold T on absolute errors (default 0.1)')
parser.add_argument(
    "--csv-path",
    type=str,
    default="results",
    dest="csv_path",
    help="Location to save results in csv format",
)
parser.add_argument("--silent", "-s", action="store_true", help="Run silently")
parser.add_argument("--verbose", "-v", action="store_true", help="Run verbose")
parser.add_argument(
    "--norm",
    type=str,
    choices=["Linfty", "L2", "L5", "LogExp"],
    default="L2",
    help="Norm to minimize if using the optimization paradigm (default: L2)",
)

parser.add_argument(
    "--categorical-consistency",
    action="store_true",
    help="Enforce consistency categorical variables",
)
parser.add_argument(
    "--measure-gen", action="store_true", help="Measure Generalization properties"
)

parser.add_argument(
    "--oversamples",
    type=str,
    default=None,
    help="comma separated values of oversamling rates (default None)",
)

args = parser.parse_args()
if args.silent and args.verbose:
    raise ValueError(
        "You cannot choose both --silent and --verbose. These are conflicting options. Choose at most one"
    )

if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(parser.format_values())
elif not args.silent:
    logging.basicConfig(level=logging.INFO)

if args.all_queries:
    logging.warning(
        "--all-queries option has been chosen. No selection of top queries will occur"
    )

if args.save_fig:
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)

    with open(os.path.join(args.destination, "config.txt"), "w") as output_config_file:
        output_config_file.write(parser.format_values())

key = random.PRNGKey(args.seed)

dataset = data_sources[args.data_source](
    args.read_file, args.filepath, args.use_subset, args.n, args.d
)
D = np.asarray(dataset.get_dataset())

# update dataset shape
args.n, args.d = D.shape

# default delta if not specified
if args.delta is None:
    args.delta = 1 / args.n ** 2

stat_module = __import__(args.statistic_module)

# First select random k-way marginals from the dataset
kway_attrs = dataset.randomKway(num_kways=args.workload, k=args.k)
kway_compact_queries, _ = dataset.get_queries(kway_attrs)
all_statistic_fn = stat_module.preserve_statistic(kway_compact_queries)
true_statistics = all_statistic_fn(D)

projection_interval = ProjectionInterval(*args.project) if args.project else None

args.epochs = (
    min(args.epochs, np.ceil(len(true_statistics) / args.top_q).astype(np.int32))
    if not args.all_queries
    else 1
)

if args.all_queries:
    # ensure consistency w/ top_q for one-shot case (is this correct?)
    args.top_q = len(true_statistics)

# Initial analysis
print("Number of queries: {}".format(len(true_statistics)))
print("Number of epochs: {}".format(args.epochs))

if args.categorical_consistency:
    print("Categorical consistency")
    feats_csum = np.array([0] + list(dataset.domain.values())).cumsum()
    feats_idx = [
        list(range(feats_csum[i], feats_csum[i + 1]))
        for i in range(len(feats_csum) - 1)
    ]
else:
    feats_idx = None

# Set up the algorithm configuration
algorithm_configuration = RAPConfiguration(
    num_points=args.n,
    num_generated_points=args.n_prime,
    num_dimensions=args.d,
    statistic_function=all_statistic_fn,
    preserve_subset_statistic=stat_module.preserve_subset_statistic,
    get_queries=dataset.get_queries,
    get_sensitivity=stat_module.get_sensitivity,
    verbose=args.verbose,
    silent=args.silent,
    epochs=args.epochs,
    iterations=args.iterations,
    epsilon=args.epsilon,
    delta=args.delta,
    norm=Norm(args.norm),
    projection_interval=projection_interval,
    optimizer_learning_rate=args.learning_rate,
    lambda_l1=args.lambda_l1,
    k=args.k,
    top_q=args.top_q,
    use_all_queries=args.all_queries,
    rap_stopping_condition=args.stopping_condition,
    initialize_binomial=args.initialize_binomial,
    feats_idx=feats_idx,
)

key, subkey = random.split(key)
rap = RAP(algorithm_configuration, key=key)
# growing number of sanitized statistics to preserve
key, subkey = random.split(subkey)
rap.train(D, kway_attrs, key)

all_synth_statistics = all_statistic_fn(rap.D_prime)

print("True statistics:")
print(true_statistics)

print("Synthetic statistics:")
print(all_synth_statistics)

print("Total number of queries:", len(true_statistics))

max_base = np.max(np.absolute(true_statistics - np.zeros(true_statistics.shape)))
l1_base = np.linalg.norm(true_statistics - np.zeros(true_statistics.shape), ord=1)
l2_base = np.linalg.norm(true_statistics - np.zeros(true_statistics.shape), ord=2)
print("Baseline max abs error", max_base)
print("Baseline L1 error", l1_base)
print("Baseline L2 error", l2_base)

max_final = np.max(np.absolute(true_statistics - all_synth_statistics))
l1_final = np.linalg.norm(true_statistics - all_synth_statistics, ord=1)
l2_final = np.linalg.norm(true_statistics - all_synth_statistics, ord=2)
print("Final max abs error", max_final)
print("Final L1 error", l1_final)
print("Final L2 error", l2_final)

if args.categorical_consistency:
    # Check loss of accuracy when randomized_rounding is applied to get rap.Dprime_ohe dataset
    if args.oversamples:
        oversampling_errors = {}
        for oversample in args.oversamples.split(","):
            print("Oversample rate:", oversample)
            max_final_ohe, l2_final_ohe = [], []
            repetitions = 10
            for rep in range(repetitions):
                Dprime_ohe = rap.generate_rounded_dataset(
                    key, oversample=int(oversample)
                )

                all_synth_statistics_ohe = all_statistic_fn(Dprime_ohe)

                max_final_ohe += [
                    np.max(np.absolute(true_statistics - all_synth_statistics_ohe))
                ]
                l2_final_ohe += [
                    np.linalg.norm(true_statistics - all_synth_statistics_ohe, ord=2)
                ]

            max_final_ohe = np.array(max_final_ohe)
            l2_final_ohe = np.array(l2_final_ohe)

            oversampling_errors[oversample] = [
                (
                    "max_error",
                    float(np.mean(max_final_ohe)),
                    float(np.std(max_final_ohe)),
                ),
                ("l2_error", float(np.mean(l2_final_ohe)), float(np.std(l2_final_ohe))),
            ]
        print(oversampling_errors)
        pd.DataFrame.from_dict(oversampling_errors, orient="index").to_csv(
            os.path.join(
                args.csv_path,
                "rounded_oversample_errors_{}_{}_{}_eps{}_T{}_K{}.csv".format(
                    args.data_source,
                    args.workload,
                    args.k,
                    args.epsilon,
                    args.epochs,
                    args.top_q,
                ),
            ),
        )
        max_final_ohe = np.mean(max_final_ohe)
        l2_final_ohe = np.mean(l2_final_ohe)

    else:
        Dprime_ohe = rap.generate_rounded_dataset(key)
        all_synth_statistics_ohe = all_statistic_fn(Dprime_ohe)
        max_final_ohe = np.max(np.absolute(true_statistics - all_synth_statistics_ohe))
        l1_final_ohe = np.linalg.norm(
            true_statistics - all_synth_statistics_ohe, ord=1
        ) / float(args.workload)

        l2_final_ohe = np.linalg.norm(true_statistics - all_synth_statistics_ohe, ord=2)
        print("\tFinal rounded max abs error", max_final_ohe)
        print("\tFinal rounded L1 error", l1_final_ohe)
        print("\tFinal rounded L2 error", l2_final_ohe)

if args.measure_gen:
    # Generalization
    # Second set of k-way marginals for generalization
    kway_attrs_2 = dataset.randomKway(num_kways=args.workload, k=args.k, seed=1)
    kway_compact_queries_2, _ = dataset.get_queries(kway_attrs_2)
    all_statistic_fn_2 = stat_module.preserve_statistic(kway_compact_queries_2)
    true_statistics_2 = all_statistic_fn_2(D)

    all_synth_statistics_2 = all_statistic_fn_2(rap.D_prime)
    max_final_2 = np.max(np.absolute(true_statistics_2 - all_synth_statistics_2))
    l1_final_2 = np.linalg.norm(true_statistics_2 - all_synth_statistics_2, ord=1)
    l2_final_2 = np.linalg.norm(true_statistics_2 - all_synth_statistics_2, ord=2)
    max_base_2 = np.max(
        np.absolute(true_statistics_2 - np.zeros(true_statistics_2.shape))
    )
    l2_base_2 = np.linalg.norm(
        true_statistics_2 - np.zeros(true_statistics_2.shape), ord=2
    )
    print("Final max abs gen error", max_final_2)
    print("Final L1 gen error", l1_final_2)
    print("Final L2 gen error", l2_final_2)

if args.csv_path:

    os.makedirs(args.csv_path, exist_ok=True)

    names = [
        "epsilon",
        "max_error",
        "l2_error",
        "max_base",
        "l2_base",
        "npoints",
        "norm",
        "l1_reg",
        "top_q",
        "epochs",
    ]
    res = [
        args.epsilon,
        max_final,
        l2_final,
        max_base,
        l2_base,
        args.n_prime,
        args.norm,
        args.lambda_l1,
        args.top_q,
        args.epochs,
    ]

    fn_prefix = ""

    if args.measure_gen:
        names += ["max_base_2", "l2_base_2", "max_gen", "l2_gen"]
        res += [max_base_2, l2_base_2, max_final_2, l2_final_2]
        fn_prefix += "gen_"

    if args.categorical_consistency:
        names += ["max_final_ohe", "l2_final_ohe", "l1_final_ohe"]
        res += [max_final_ohe, l2_final_ohe, l1_final_ohe]
        fn_prefix += "rounded_"

    df_res = pd.DataFrame([res], columns=names)

    file_name = os.path.join(
        args.csv_path,
        fn_prefix
        + "_adaptive_{}_{}_{}.csv".format(args.data_source, args.workload, args.k),
    )

    print("Saving ", file_name)

    if os.path.exists(file_name):
        dfprev = pd.read_csv(file_name)
        df_res = df_res.append(dfprev, sort=False)

    df_res.sort_values(
        by=["epsilon", "norm", "npoints", "l1_reg"], ascending=False
    ).to_csv(file_name, index=False)

    pd.DataFrame(rap.D_prime).to_csv(
        os.path.join(
            args.csv_path,
            "synthetic_data_{}_{}_{}.csv".format(
                args.data_source, args.workload, args.k
            ),
        ),
        index=False,
    )

    if args.categorical_consistency:
        Dprime_cat = ohe_to_categorical(Dprime_ohe, feats_idx)
        pd.DataFrame(data=Dprime_cat, columns=list(dataset.domain.keys())).to_csv(
            os.path.join(
                args.csv_path,
                "rounded_synthetic_data_{}_{}_{}_eps{}_T{}_K{}.csv".format(
                    args.data_source,
                    args.workload,
                    args.k,
                    args.epsilon,
                    args.epochs,
                    args.top_q,
                ),
            ),
            index=False,
        )
