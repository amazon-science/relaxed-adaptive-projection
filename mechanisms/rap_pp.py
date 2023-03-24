import os

import numpy as np
from jax import devices
from jax import random

cpu = devices("cpu")[0]
import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, List

from jax import jit
from jax import numpy as jnp
from jax import random, value_and_grad
from jax.example_libraries import optimizers

from dataloading.dataset import Dataset
from mechanisms.mechanism_base import BaseConfiguration, BaseMechanism
from mechanisms.util import initialize_synthetic_dataset, sparsemax_project
from privacy_budget_tracking import privacy_util

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@dataclass
class RAPppConfiguration(BaseConfiguration):
    optimizer_learning_rate: List[float] = field(default_factory=lambda: [0.01])
    iterations: List[int] = field(default_factory=lambda: [30])
    rap_percent_stopping_condition: List[float] = field(
        default_factory=lambda: [0.0001]
    )
    clip_grad: float = 0.1
    sigmoid_0: List[float] = field(default_factory=lambda: [2])
    sigmoid_doubles: list = field(default_factory=lambda: [10])
    loss_type: list = field(default_factory=lambda: [2])
    truncate_sigmoid: list = field(default_factory=lambda: [True])

    top_q: int = 1
    get_dp_select_epochs: Callable = (
        lambda domain: len(domain.attrs) - 1
    )  # returns select epochs as function of domain
    get_privacy_budget_weight: Callable = (
        lambda domain: 1
    )  # privacy budget used for this class


def print_bold(s, end="\n"):
    BOLD = "\033[1m"
    END = "\033[0m"
    print(f"{BOLD}{s}{END}", end=end)


class LogTime:
    def __init__(self):
        self.stime = time.time()

    def log_time(self, msg=""):
        print_bold(f"{msg}({time.time() - self.stime:.4f}s)")
        self.stime = time.time()


class ManageStats:
    def __init__(self, statistics, D, SUB_QUERIES_SIZE):
        self.stat = statistics
        self.SUB_QUERIES_SIZE = SUB_QUERIES_SIZE

        self.num_queries = statistics.get_num_queries()
        print(f"num_queries={self.num_queries}")

        self.all_stat_idx = jnp.arange(statistics.get_num_queries())

        print_bold(f"Computing statistic function:", end="")
        stime = time.time()
        self.jit_true_stat_fn = jit(
            statistics.get_exact_statistics_fn(
                queries_size=statistics.get_num_queries()
            )
        )
        self.true_D_stats = self.jit_true_stat_fn(
            self.all_stat_idx, D
        ).block_until_ready()

        print_bold(f"...done! ({time.time() - stime:.2f}).")

        self.jit_sub_exact_statistics_fn = jit(
            statistics.get_exact_statistics_fn(queries_size=SUB_QUERIES_SIZE)
        )
        self.jit_sub_diff_statistics_fn = jit(
            statistics.get_differentiable_statistics_fn(queries_size=SUB_QUERIES_SIZE)
        )

    def compute_all_true_stats(self, D_temp):
        return self.jit_true_stat_fn(self.all_stat_idx, D_temp)


class RAPpp(BaseMechanism):
    def get_loss_functions(self, stat_holder: ManageStats):
        exact_stat_fn = stat_holder.jit_sub_exact_statistics_fn
        differentialble_stat_fn = stat_holder.jit_sub_diff_statistics_fn

        @jit
        def progress_loss_fn(D_prime, queries_idx, private_target_answers):
            stats1 = exact_stat_fn(queries_idx, D_prime)
            error1 = private_target_answers - stats1
            loss_1 = jnp.linalg.norm(error1, ord=2)
            return loss_1

        @jit
        def loss_fn(D_prime, sigmoid_param, queries_idx, private_target_answers):
            stats1 = differentialble_stat_fn(queries_idx, D_prime, sigmoid_param)
            error1 = private_target_answers - stats1
            loss_1 = jnp.linalg.norm(error1, ord=2)
            return loss_1

        return progress_loss_fn, loss_fn

    def get_update_fn(self, domain, loss_fn):
        opt_init, opt_update, get_params = optimizers.adam(lambda x: x)

        feats_csum = jnp.array([0] + list(domain.shape)).cumsum()
        feats_idx = [
            list(range(feats_csum[i], feats_csum[i + 1]))
            for i in range(len(feats_csum) - 1)
        ]

        @jit
        def update_fn(
            state, sigmoid_param, opt_lr, queries_idx, private_target_answers
        ):
            """Compute the gradient and update the parameters"""
            D_prime = get_params(state)
            value, grads = value_and_grad(loss_fn, argnums=0)(
                D_prime, sigmoid_param, queries_idx, private_target_answers
            )
            state = opt_update(opt_lr, grads, state)

            unpacked_state = optimizers.unpack_optimizer_state(state)
            new_D_prime = unpacked_state.subtree[0]
            new_D_prime = sparsemax_project(new_D_prime, feats_idx)
            new_D_prime = self._clip_array(new_D_prime)
            unpacked_state.subtree = (
                new_D_prime,
                unpacked_state.subtree[1],
                unpacked_state.subtree[2],
            )
            updated_state = optimizers.pack_optimizer_state(unpacked_state)

            return updated_state, get_params(updated_state), value

        return update_fn

    def initialize(self, dataset: Dataset, seed: int):
        """This function is called once before _train().
        The purpose of this is to save compilation time.
        Call once for each dataset and then call _train() multiple times for different seeds and epsilon values.
        1) Initializes the privacy weight for each statistic module.
        2) Initializes compilation of jax.jit statistic functions. ()
        """
        super().initialize(dataset, seed)
        D = jnp.asarray(dataset.get_dataset())
        self.num_points = D.shape[0]
        self.domain = dataset.domain
        args = self.args_list
        self.NUM_STATS = len(self.statistics)
        self.STAT = {}
        self.SELECT_EPOCHS = {}
        self.PRIVACY_WEIGHTS = {}
        self.TOP_q = {}

        # Each statistics module defines a weight for how much privacy budget to use.
        TOTAL_PRIV_WEIGHTS = np.sum(
            [
                args[k].get_privacy_budget_weight(self.domain)
                for k in range(self.NUM_STATS)
            ]
        )
        for k in range(self.NUM_STATS):
            args_k = args[k]
            # Set up privacy parameters for the k-th query class

            # Compute privacy weight for the k-th statistic module.
            self.SELECT_EPOCHS[k] = args_k.get_dp_select_epochs(dataset.domain)
            self.PRIVACY_WEIGHTS[k] = (
                args_k.get_privacy_budget_weight(self.domain) / TOTAL_PRIV_WEIGHTS
            )
            self.TOP_q[k] = args_k.top_q

            # Start k-th statistics. Compute the size.
            SUB_QUERIES_SIZE = max(
                1, self.TOP_q[k] * self.SELECT_EPOCHS[k]
            )  # Total number of queries that will be sampled from the k-th query class
            print(f"SUB_QUERIES_SIZE={SUB_QUERIES_SIZE}")

            stat_k = ManageStats(self.statistics[k], D, SUB_QUERIES_SIZE)
            self.STAT[k] = stat_k

            print(f"self.SELECT_EPOCHS[{k}]={self.SELECT_EPOCHS[k]}")
            print(f"self.PRIVACY_WEIGHTS[{k}]={self.PRIVACY_WEIGHTS[k]}")
            print(f"self.TOP_q[{k}]={self.TOP_q[k]}")

    def _train(
        self, args: list, num_generated_points: int, epsilon, seed, debug_fn=None
    ):
        logtime = LogTime()
        data_dimension = sum(self.domain.shape)
        delta = 1 / self.num_points**2

        key = random.PRNGKey(seed)
        init_key = key.copy()
        np.random.seed(seed)

        D_prime = jnp.asarray(
            initialize_synthetic_dataset(init_key, num_generated_points, data_dimension)
        )

        QUERY_HISTORY = {}

        self.private_selected_queries_idx = {}
        self.private_statistics = {}

        for k in range(self.NUM_STATS):
            QUERY_HISTORY[k] = []
            self.private_selected_queries_idx[k] = (
                -jnp.ones(self.STAT[k].SUB_QUERIES_SIZE).astype(int).reshape(-1)
            )

        epoch_type = jnp.concatenate(
            [K * np.ones(self.SELECT_EPOCHS[K]) for K in range(len(args))]
        ).astype(int)
        epoch_type = np.random.permutation(epoch_type)

        rho_total = privacy_util.get_zcdp_parameter(epsilon, delta)

        rho_per_select_epoch = {}

        for k in range(self.NUM_STATS):
            budget_for_k_stat = self.PRIVACY_WEIGHTS[k]
            rho_per_select_epoch[k] = (
                budget_for_k_stat * rho_total / self.SELECT_EPOCHS[k]
            )
            print(f"budget_for_k_stat[{k}] = {budget_for_k_stat:.5f}")
            print(
                f"rho_per_select_epoch[{k}] = {rho_per_select_epoch[k]:.5f} out of rho_total={rho_total:.5f}"
            )

        PROGRESS_LOSS = {}
        UPT_fn = {}
        for k in range(self.NUM_STATS):
            progress_loss_k, loss_k = self.get_loss_functions(self.STAT[k])
            PROGRESS_LOSS[k] = progress_loss_k
            UPT_fn[k] = self.get_update_fn(self.domain, loss_k)

        stime = time.time()
        for epoch, K in enumerate(epoch_type):
            key, subkey = random.split(key)

            args_k = args[K]
            # Set up privacy parameters for the k-th query class
            top_q = self.TOP_q[K]
            rho_RNM = (1 / 2) * rho_per_select_epoch[K]
            rho_GaussianMech = (1 / 2) * rho_per_select_epoch[K]
            module_sensitivity = self.STAT[K].stat.get_sensitivity()
            sensitivity = module_sensitivity * np.sqrt(top_q) / self.num_points  #

            true_statistics_temp = self.STAT[K].true_D_stats

            # Compute sync statistics for the k-th query class
            sync_statistics_temp = self.STAT[K].compute_all_true_stats(D_prime)

            l_inf_error = np.max(np.abs(true_statistics_temp - sync_statistics_temp))

            # RNM:
            query_errors = np.linalg.norm(
                true_statistics_temp - sync_statistics_temp, ord=1, axis=1
            )

            bias = self.STAT[K].stat.marginal_size / self.num_points
            query_errors_penalty = query_errors - bias
            selected_indices = privacy_util.select_noisy_q(
                query_errors_penalty,
                np.array(QUERY_HISTORY[K]),
                # Do not resample these
                top_q,
                query_select_budget=rho_RNM,
                sensitivity=sensitivity,
            )

            # Save selected queries
            epoch_max_error = 0
            for sele_id in selected_indices:
                sele_id = int(sele_id)
                if K == 0:
                    mar_size = self.STAT[K].stat.marginal_size[sele_id]
                    print(
                        # f"{self.STAT[K].stat.workload[sele_id]}. size={mar_size}, "
                        f"query_errors = {query_errors[sele_id]:.4f}, "
                        # f'query_errors_inf = {query_errors_inf[sele_id]:.4f}'
                    )
                epoch_max_error = max(epoch_max_error, query_errors[sele_id])
                QUERY_HISTORY[K].append(sele_id)

            # Get true stats from the k-th statistic class
            self.private_selected_queries_idx[K] = (
                jnp.concatenate(
                    [
                        jnp.array(QUERY_HISTORY[K]),
                        -jnp.ones(
                            self.STAT[K].SUB_QUERIES_SIZE - len(QUERY_HISTORY[K])
                        ),
                    ]
                )
                .astype(int)
                .reshape(-1)
            )

            # Get the set of true statistics for this epoch and add noise.
            selected_true_statistics = jnp.vstack(
                [self.STAT[K].true_D_stats[i, :] for i in QUERY_HISTORY[K]]
            )

            private_statistics = privacy_util.gaussian_mechanism(
                subkey,
                selected_true_statistics,
                query_select_budget=rho_GaussianMech,
                sensitivity=sensitivity,
            )
            private_statistics = jnp.asarray(private_statistics)

            # Statistic answers must be between [0, 1]
            private_statistics = jnp.clip(private_statistics, 0, 1)

            # Add zeros at the end of this object to make shape constant
            h, w = private_statistics.shape
            self.private_statistics[K] = jnp.concatenate(
                [private_statistics, jnp.zeros((self.STAT[K].SUB_QUERIES_SIZE - h, w))]
            )

            # Define the loss and update functions for this epoch. note that this is using jax precompiled functions.
            progress_fn = lambda D_prime: PROGRESS_LOSS[K](
                D_prime,
                self.private_selected_queries_idx[K],
                self.private_statistics[K],
            )
            update_fn = lambda opt_state, sigmoid_param, opt_lr: UPT_fn[K](
                opt_state,
                sigmoid_param,
                opt_lr,
                self.private_selected_queries_idx[K],
                self.private_statistics[K],
            )

            # Run projection mechanism.
            progress_loss_begin = progress_fn(D_prime)
            D_prime = self.projection_mechanism(
                args_k, D_prime, progress_loss_fn=progress_fn, update_fn=update_fn
            )

            progress_loss_final = progress_fn(D_prime)
            gaussian_error = private_statistics - selected_true_statistics

            print_bold(f"==> Epoch {epoch} type=[{K}]:", end=" ")
            print_bold(
                f"Select Max error={epoch_max_error:.3f}, Max l1_Norm Error ={np.max(query_errors):.3f}, "
                f"Ave l1_Norm Error={np.linalg.norm(query_errors, ord=1) / query_errors.shape[0]:.5f}.",
                end=" ",
            )
            print_bold(f"l_inf_error={l_inf_error:.4f}.", end=" ")
            print_bold(
                f"Gau: max={jnp.max(jnp.abs(gaussian_error)):.3f}, l2={jnp.linalg.norm(gaussian_error, ord=2):.4f}.",
                end=" ",
            )
            print_bold(
                f"query[{K}]: Proj Loss: start={progress_loss_begin:.4f}, end={progress_loss_final:.4f}. ({time.time() - stime:.2f}s)"
            )
            stime = time.time()

            if epoch > 25 and (epoch + 1) % 10 == 0 and debug_fn is not None:
                debug_fn(D_prime)

        for K in range(self.NUM_STATS):
            sync_statistics_temp = self.STAT[K].compute_all_true_stats(D_prime)
            query_errors = np.linalg.norm(
                self.STAT[K].true_D_stats - sync_statistics_temp, ord=2, axis=1
            )
            print_bold(
                f"Final Error {K}: Max={np.max(query_errors):.3f}, "
                f"Ave={np.linalg.norm(query_errors, ord=1) / query_errors.shape[0]:.5f}."
            )
        self.D_prime = D_prime

    def projection_mechanism(
        self, args: RAPppConfiguration, D_prime_init, progress_loss_fn, update_fn
    ):
        D_prime_l2_loss_min = np.inf

        D_prime = None
        for loss_type in args.loss_type:
            opt_init, opt_update, get_params = optimizers.adam(lambda x: x)
            """ Loss function for a single row of D_prime """

            for opt_lr, iters, rp_stop, sigmoid_0, sig_doubles in itertools.product(
                args.optimizer_learning_rate,
                args.iterations,
                args.rap_percent_stopping_condition,
                args.sigmoid_0,
                args.sigmoid_doubles,
            ):
                D_prime = D_prime_init.copy()

                """ Use SGD with corresponding parameters to find a D' that minimized the loss function using the corresponding parameters  """
                if args.debug:
                    print(
                        f"\n\ndebug:"
                        f"learning rate={opt_lr}, "
                        f"iters={iters}, "
                        f"sigmoid_0={sigmoid_0}, "
                        f"loss_type={loss_type}, "
                        f"sig_doubles={sig_doubles}, "
                    )
                D_prime = self.run_sgd_mixed(
                    D_prime,
                    opt_init,
                    progress_loss_fn,
                    update_fn,
                    opt_lr,
                    iters,
                    rp_stop,
                    sigmoid_0,
                    sig_doubles,
                    args.debug,
                )
                D_prime_progress_loss = progress_loss_fn(D_prime)

                D_prime, D_prime_l2_loss_min = (
                    (D_prime, D_prime_progress_loss)
                    if D_prime_progress_loss < D_prime_l2_loss_min
                    else (D_prime, D_prime_l2_loss_min)
                )

        return D_prime

    def run_sgd_mixed(
        self,
        D_prime,
        opt_init,
        progress_loss_fn,
        update_fn,
        opt_lr,
        iters,
        rp_sto,
        sigmoid_0,
        sigmoid_double,
        debug,
    ):
        """Get update function using stochastic batching"""
        D_prime_best = D_prime
        best_loss = progress_loss_fn(D_prime_best)  # ~ 0
        epoch_previous_loss = best_loss
        opt_state = opt_init(D_prime)
        stime = time.time()
        timer = LogTime()

        for e in range(iters + 1):
            """Train each split"""
            sig_counter = 0
            sigmoid_param = sigmoid_0
            loss_list = [jnp.inf]
            opt_lr_epoch = opt_lr / 2**e
            for mini_e in range(2000):
                opt_state, D_prime, loss = update_fn(
                    opt_state, sigmoid_param, opt_lr_epoch
                )

                epoch_loss = float(progress_loss_fn(D_prime))  # 0.00001 > 0
                loss_list.append(epoch_loss)

                """ Stop early and Save best D_prime """
                D_prime_best, best_loss = (
                    (D_prime, epoch_loss)
                    if epoch_loss < best_loss
                    else (D_prime_best, best_loss)
                )

                if debug:
                    print(
                        f"{e:<3}, mini_e={mini_e:<4}: opt_lr_epoch={opt_lr_epoch}."
                        f"\tl2-loss-diff={loss:.5f}, progress_loss={epoch_loss:.5f}, "
                        f"sig_counter={sig_counter},\ttime={time.time() - stime:.3f}s"
                    )
                if len(loss_list) > 10:
                    ave_last = loss_list[-5]
                    percent_change = (ave_last - epoch_loss) / (ave_last + 1e-9)
                    if percent_change < 0.0001:
                        if debug:
                            print(
                                f"ave_last={ave_last:.4f}, epoch_loss={epoch_loss:.4f}"
                            )
                        sigmoid_param = 2 * sigmoid_param
                        sig_counter += 1
                        loss_list = []
                        if sig_counter > sigmoid_double:
                            break

            # """ Stop early """
            epoch_loss = float(progress_loss_fn(D_prime))
            epoch_loss_change = (epoch_previous_loss - epoch_loss) / (
                epoch_previous_loss + 1e-9
            )
            epoch_loss_change = np.abs(epoch_loss_change)
            if debug:
                print(f"epoch_loss_change={epoch_loss_change:.4f}")
            if (
                rp_sto > 0
                and epoch_previous_loss is not None
                and epoch_loss_change < rp_sto
            ):
                break
            epoch_previous_loss = epoch_loss

        return D_prime_best
