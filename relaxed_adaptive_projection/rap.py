import time
import logging
from typing import Tuple, Any, Callable

import numpy as np_orig
from jax import numpy as np, random, jit, value_and_grad
from jax.experimental import optimizers

from privacy_budget_tracking import zCDPTracker
from utils_data import sparsemax_project, randomized_rounding
from .constants import SyntheticInitializationOptions, norm_mapping, Norm
from .rap_configuration import RAPConfiguration


class RAP:
    def __init__(self, args: RAPConfiguration, key: np.DeviceArray):
        self.args = args

        self.start_time = time.time()
        # Initialize the synthetic dataset
        self.D_prime = self.__initialize_synthetic_dataset(key)

        self.statistics_l1 = []
        self.statistics_max = []
        self.means_l1 = []
        self.means_max = []
        self.max_errors = []
        self.l2_errors = []
        self.losses = []

        self.tracker = zCDPTracker(
            args.epsilon, args.delta, args.epochs, args.top_q, args.num_points
        )

        self.feats_idx = args.feats_idx

        if self.args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        elif not self.args.silent:
            logging.basicConfig(level=logging.INFO)

    def __log_progress(self, iteration: int = -1) -> None:
        """
        Log the latest iteration statistic values unless an iteration provide
        :param iteration:
        """
        log_iteration = len(self.statistics_l1) - 1 if iteration < 0 else iteration
        statistic_l1 = self.statistics_l1[iteration]
        statistic_max = self.statistics_max[iteration]
        mean_l1 = self.means_l1[iteration]
        mean_max = self.means_max[iteration]
        loss = self.losses[iteration]
        l2_error = self.l2_errors[iteration]
        max_error = self.max_errors[iteration]

        logging.info(
            "Iteration: {}; L1 of statistics: {}; L-Infty of statistics: {}; Means of L1: {}; Max of mean: {}".format(
                log_iteration, statistic_l1, statistic_max, mean_l1, mean_max
            )
        )

        logging.info(
            "Loss: {:.5f}, L2 Error {:.5f} Max Error {:.5f}".format(
                loss, l2_error, max_error
            )
        )

    def __compute_initial_dataset(
        self, selection: SyntheticInitializationOptions, key: np.DeviceArray
    ) -> np.DeviceArray:
        """
        Function that computes D_prime based on input
        :param selection: the type of synthetic data initialization
        :param: key: key to generate random numbers with
        :return: initial hypothesis of synthetic data
        """
        shape = (self.args.num_generated_points, self.args.num_dimensions)
        random_initial = random.uniform(key=key, shape=shape)

        if selection is SyntheticInitializationOptions.RANDOM:
            return 2 * (random_initial - 0.5)
        elif selection is SyntheticInitializationOptions.RANDOM_INTERVAL:
            interval = self.args.projection_interval
            return (
                (interval.projection_max - interval.projection_min) * random_initial
            ) + interval.projection_min
        elif selection is SyntheticInitializationOptions.RANDOM_BINOMIAL:
            return np_orig.array(
                [
                    np_orig.random.binomial(1, p=self.args.probs)
                    for _ in range(self.args.num_generated_points)
                ],
                dtype=float,
            )
        else:
            raise ValueError(
                "Supported selections are ",
                [
                    member.value
                    for _, member in SyntheticInitializationOptions.__members__.items()
                ],
            )

    def __initialize_synthetic_dataset(self, key: np.DeviceArray):
        """
        Function that
        :param key: key to generate random numbers with
        :return:
        """
        if self.args.projection_interval:
            # If we are projecting into [a,b], start with a dataset in range.
            interval = self.args.projection_interval
            if len(interval) != 2 or interval.projection_max <= interval.projection_min:
                raise ValueError(
                    "Must input interval in the form '--project a b' to project into [a,b], b>a"
                )
            return self.__compute_initial_dataset(
                SyntheticInitializationOptions.RANDOM_INTERVAL, key
            )
        else:
            if self.args.initialize_binomial:
                return self.__compute_initial_dataset(
                    SyntheticInitializationOptions.RANDOM_BINOMIAL, key
                )
            else:
                return self.__compute_initial_dataset(
                    SyntheticInitializationOptions.RANDOM, key
                )

    def __compute_composition(self) -> Tuple[float, float]:
        # advanced composition
        epsilon_p = self.args.epsilon / np.sqrt(
            2 * (2 * self.args.epochs) * np.log((2 * self.args.num_points) ** 2)
        )
        delta_p = (self.args.num_points ** -2 - (2 * self.args.num_points) ** -2) / (
            2 * self.args.epochs
        )
        return epsilon_p, delta_p

    def __jit_loss_fn(
        self, statistic_fn: Callable[[np.DeviceArray], np.DeviceArray]
    ) -> Callable[[np.DeviceArray, np.DeviceArray], np.DeviceArray]:

        ord_norm = norm_mapping[self.args.norm]

        @jit
        def compute_loss_fn(
            synthetic_dataset: np.DeviceArray, target_statistics: np.DeviceArray
        ) -> np.DeviceArray:
            if self.args.norm is Norm.LOG_EXP:
                return np.log(
                    np.exp(statistic_fn(synthetic_dataset) - target_statistics).sum()
                ) + self.args.lambda_l1 * np.linalg.norm(synthetic_dataset, 1)
            else:
                return np.linalg.norm(
                    statistic_fn(synthetic_dataset) - target_statistics, ord=ord_norm
                ) + self.args.lambda_l1 * np.linalg.norm(synthetic_dataset, 1)

        return compute_loss_fn

    def __get_update_function(
        self,
        learning_rate: float,
        optimizer: Callable[..., optimizers.Optimizer],
        loss_fn: Callable[[np.DeviceArray, np.DeviceArray], np.DeviceArray],
    ) -> Tuple[
        Callable[[np.DeviceArray, np.DeviceArray], np.DeviceArray], np.DeviceArray
    ]:

        opt_init, opt_update, get_params = optimizer(learning_rate)
        opt_state = opt_init(self.D_prime)

        @jit
        def update(synthetic_dataset, target_statistics, state):
            """Compute the gradient and update the parameters"""
            value, grads = value_and_grad(loss_fn)(synthetic_dataset, target_statistics)
            state = opt_update(0, grads, state)
            return get_params(state), state, value

        return update, opt_state

    def __clip_array(self, array: np.DeviceArray) -> np.DeviceArray:
        if self.args.projection_interval:
            projection_min, projection_max = self.args.projection_interval
            return np.clip(array, projection_min, projection_max)
        else:
            return array

    def train(
        self, dataset: np.DeviceArray, k_way_attributes: Any, key: np.DeviceArray
    ) -> None:
        true_statistics = self.args.statistic_function(dataset)

        sanitized_queries = np.array([])
        target_statistics = np.array([])
        k_way_queries, total_queries = self.args.get_queries(k_way_attributes, N=-1)
        k_way_queries = np.asarray(k_way_queries)

        for epoch in range(self.args.epochs):
            query_errs = np.abs(
                self.args.statistic_function(self.D_prime) - true_statistics
            )
            logging.info(
                "Average error: {}, Error standard deviation: {}".format(
                    query_errs.mean(), query_errs.std()
                )
            )

            # compute zcdp budgets for epoch
            epoch_answer_budget = self.tracker.budget_per_epoch()
            epoch_select_budget = 0
            if not self.args.use_all_queries:
                epoch_select_budget = epoch_answer_budget / 2
                epoch_answer_budget /= 2

            epoch_total_budget = epoch_answer_budget + epoch_select_budget
            # sensitivity = self.args.num_points ** -1
            # epsilon_p, delta_p = self.__compute_composition()
            logging.info("zCDP budget per epoch  -- Rho: {}".format(epoch_total_budget))
            logging.info(
                "DP budget per epoch  -- Epsilon: {}, Delta: {}".format(
                    zCDPTracker.convert_zcdp_to_eps_delta(
                        epoch_total_budget, self.args.delta
                    ),
                    self.args.delta,
                )
            )

            # select new top k noisy queries to sanitize selected_idxs = numeric_sparse(queries_error,
            # set(sanitized_queries), args.top_k, args.select_T, eps_p, delta=delta_p)

            if self.args.use_all_queries:
                selected_indices = np.arange(len(k_way_queries))
            else:
                # compute budget, noise for noisy_top_k

                # selected_indices = noisy_top_k(query_errs, sanitized_queries, self.args.top_q, epsilon_p, sensitivity,
                # delta_p)
                selected_indices = self.tracker.select_noisy_q(
                    query_errs, sanitized_queries, self.args.top_q, epoch_select_budget
                )

            selected_queries = k_way_queries.take(selected_indices, axis=0)
            current_statistic_fn = self.args.preserve_subset_statistic(selected_queries)

            num_epoch_queries = len(selected_queries)
            query_answer_budget = epoch_answer_budget / num_epoch_queries

            target_statistics = np.concatenate(
                [
                    target_statistics,
                    self.tracker.gaussian_mechanism(
                        key, current_statistic_fn, dataset, query_answer_budget
                    ),
                ]
            )

            sanitized_queries = np.asarray(
                np.append(sanitized_queries, selected_indices), dtype=np.int32
            )

            curr_queries = k_way_queries[sanitized_queries]
            curr_statistic_fn = self.args.preserve_subset_statistic(
                np.asarray(curr_queries)
            )

            target_statistics = self.__clip_array(target_statistics)

            loss_fn = self.__jit_loss_fn(curr_statistic_fn)
            previous_loss = np.inf

            optimizer_learning_rate = (
                self.args.optimizer_learning_rate
            )  # * 2 ** (-epoch)
            update, opt_state = self.__get_update_function(
                optimizer_learning_rate, optimizers.adam, loss_fn
            )

            for iteration in range(self.args.iterations):

                self.D_prime, opt_state, loss = update(
                    self.D_prime, target_statistics, opt_state
                )
                if self.feats_idx:
                    self.D_prime = sparsemax_project(self.D_prime, self.feats_idx)
                    # self.D_prime = self.__clip_array(self.D_prime)
                else:
                    self.D_prime = self.__clip_array(self.D_prime)

                synthetic_statistics = curr_statistic_fn(self.D_prime)
                self.statistics_l1.append(
                    np.mean(np.absolute(target_statistics - synthetic_statistics))
                )
                self.statistics_max.append(
                    np.amax(np.absolute(target_statistics - synthetic_statistics))
                )
                self.means_l1.append(
                    np.mean(np.absolute(np.mean(dataset, 0) - np.mean(self.D_prime, 0)))
                )
                self.means_max.append(
                    np.amax(np.absolute(np.mean(dataset, 0) - np.mean(self.D_prime, 0)))
                )
                all_synth_statistics = self.args.statistic_function(self.D_prime)
                self.max_errors.append(
                    np.max(np.absolute(true_statistics - all_synth_statistics))
                )
                self.l2_errors.append(
                    np.linalg.norm(true_statistics - all_synth_statistics, ord=2)
                )
                self.losses.append(loss)

                # Stop early if we made no progress this round.
                if loss >= previous_loss - self.args.rap_stopping_condition:
                    logging.info("Stopping early at iteration {}".format(iteration))
                    self.__log_progress()
                    break

                if iteration % 100 == 0:
                    self.__log_progress()

                previous_loss = loss

        total_zcdp_used = self.tracker.get_budget_used()
        logging.info("zCDP budget used  -- Rho: {}".format(total_zcdp_used))
        logging.info(
            "DP budget used  -- Epsilon: {}, Delta: {}".format(
                zCDPTracker.convert_zcdp_to_eps_delta(total_zcdp_used, self.args.delta),
                self.args.delta,
            )
        )

    def generate_rounded_dataset(self, key, oversample=None):
        if not oversample:
            oversample = self.args.num_points // self.args.num_generated_points

        return randomized_rounding(
            D=self.D_prime, feats_idx=self.feats_idx, key=key, oversample=oversample
        )
