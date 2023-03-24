import argparse
import os

from jax import devices

cpu = devices("cpu")[0]
from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sys

sys.path.append(os.path.dirname("./"))
"""
Use this environment:
https://aws.amazon.com/releasenotes/aws-deep-learning-ami-gpu-cuda-11-4-amazon-linux-2/
"""

import run
from dataloading.data_functions.acs import *
from mechanisms.rap_pp import RAPpp, RAPppConfiguration
from modules.marginal_queries import MarginalQueryClass
from modules.random_features_linear import RandomFeaturesLinearQueryClass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="RAP++")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="privacy budget in epsilon"
    )
    parser.add_argument("--k", type=int, default=2, help="K-way marginals")
    parser.add_argument(
        "--num_random_projections",
        type=int,
        default=200_000,
        help="Number of random projections for random queries",
    )
    parser.add_argument(
        "--top_q",
        type=int,
        default=5,
        help="Number of queries with maximum errors to release privately each round",
    )
    parser.add_argument(
        "--dp_select_epochs", type=int, default=50, help="Number of epochs"
    )
    parser.add_argument(
        "--states", type=str, default="NY,CA", help="Load data from the state(s)"
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="income,coverage",
        help="Load tasks from this state(s)",
    )
    parser.add_argument(
        "--multitask",
        action="store_true",
        help="Generate synthetic data that works for multiple tasks at once",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="RAP++",
        choices=["RAP", "RAP++"],
        help="RAP or RAP++",
    )

    args = parser.parse_args()

    if args.multitask:
        datasets_fns = [get_acs_all(state=state) for state in args.states.split(",")]
    else:
        datasets_fns = [
            get_acs(state=state, target=target)
            for state in args.states.split(",")
            for target in args.targets.split(",")
        ]

    rap_args = RAPppConfiguration(
        iterations=[1],
        sigmoid_doubles=[0],
        optimizer_learning_rate=[0.003],
        top_q=1,
        get_dp_select_epochs=lambda domain: len(domain.get_cat_cols()),
        get_privacy_budget_weight=lambda domain: len(domain.get_cat_cols()),
        debug=False,
    )

    rap_args2 = RAPppConfiguration(
        iterations=[3],
        sigmoid_0=[2],
        sigmoid_doubles=[10],
        optimizer_learning_rate=[0.006],
        top_q=args.top_q,
        get_dp_select_epochs=lambda domain: args.dp_select_epochs,
        get_privacy_budget_weight=lambda domain: len(domain.get_cont_cols()),
        debug=False,
    )
    if args.algorithm == "RAP++":
        rap_linear_projection = RAPpp(
            [rap_args, rap_args2],
            [
                MarginalQueryClass(K=args.k),
                RandomFeaturesLinearQueryClass(
                    num_random_projections=args.num_random_projections,
                    max_number_rows=2000,
                ),
            ],
            name=f"RAP(Marginal&Halfspace)",
        )
    if args.algorithm == "RAP":
        rap_linear_projection = RAPpp(
            [rap_args],
            [
                MarginalQueryClass(K=args.k),
            ],
            name=f"RAP",
        )

    algorithm = [rap_linear_projection]

    override_errors_file = True
    saves_results = True

    for algo in algorithm:
        for dataset_fn in datasets_fns:
            run.run_experiment(
                algo,
                dataset_fn,
                epsilons_list=[args.epsilon],
                algorithm_seed=[args.seed],
                params=str((args.top_q, args.dp_select_epochs)),
                save_sycn_data=True,
                run_ml_eval=False,
            )
