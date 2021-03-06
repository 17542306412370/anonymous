import copy
import logging
import os
import time
import argparse

import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTorchPolicy

from submission.algos.lola import (
    train_cg_tune_class_API,
)
from submission.envs import (
    vectorized_coin_game,
    vectorized_ab_coin_game,
    matrix_sequential_social_dilemma,
)
from submission.utils import (
    policy,
    log,
    cross_play,
    exp_analysis,
    callbacks,
    aggregate_and_plot_tensorboard_data,
)
from submission.utils.plot import PlotConfig
from submission.base_game_experiments import lola_exact_official

logger = logging.getLogger(__name__)


def main(debug: bool, env=None):
    """
    Train several LOLA_PG pairs of agent on the selected environment and
    plot their performances in self-play and cross-play.

    :param debug: selection of debug mode using less compute
    :param env: option to overwrite the env selection
    """
    tune_hparams = _get_hyperparameters(debug, env)

    if tune_hparams["load_plot_data"] is None:
        ray.init(num_cpus=10, num_gpus=0, local_mode=debug)
        experiment_analysis_per_welfare = _train(tune_hparams)
    else:
        experiment_analysis_per_welfare = None

    _evaluate(tune_hparams, debug, experiment_analysis_per_welfare)
    ray.shutdown()


def _get_hyperparameters(debug, env=None):

    train_n_replicates = None

    if not debug:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--env",
            type=str,
            choices=["CoinGame", "ABCoinGame"],
            help="Env to use.",
        )
        parser.add_argument("--train_n_replicates", type=int)
        args = parser.parse_args()
        args = args.__dict__
        if "env" in args.keys():
            env = args["env"]
        if "train_n_replicates" in args.keys():
            train_n_replicates = args["train_n_replicates"]

    if train_n_replicates is None:
        train_n_replicates = 2 if debug else 40
    timestamp = int(time.time())
    seeds = [seed + timestamp for seed in list(range(train_n_replicates))]

    exp_name, _ = log.log_in_current_day_dir("LOLA_PG")

    tune_hparams = {
        "debug": debug,
        "exp_name": exp_name,
        "train_n_replicates": train_n_replicates,
        "classify_into_welfare_fn": True,
        # Print metrics
        "load_plot_data": None,
        # Example: "load_plot_data": ".../SelfAndCrossPlay_save.p",
        #
        # "env_name": "CoinGame" if env is None else env,
        "env_name": "ABCoinGame" if env is None else env,
        # "remove_trials_below_speed": False,
        "remove_trials_below_speed": 0.2,
        "remove_trials_below_speed_for_both": True,
        "pseudo": False,
        "grid_size": 3,
        "lola_update": True,
        "opp_model": False,
        "mem_efficient": True,
        "lr_correction": 1,
        "global_lr_divider": 1 / 10,
        "simple_net": True,
        "hidden": 32,
        "reg": 0,
        "set_zero": 0,
        "exact": False,
        "warmup": 1,
        "seed": tune.grid_search(seeds),
        "changed_config": False,
        "ac_lr": 1.0,
        "summary_len": 1,
        "use_MAE": False,
        "use_toolbox_env": True,
        "clip_loss_norm": False,
        "clip_lola_update_norm": False,
        "clip_lola_correction_norm": 3.0,
        "clip_lola_actor_norm": 10.0,
        "lr_decay": True,
        "correction_reward_baseline_per_step": False,
        "use_critic": False,
        "plot_keys": [
            "reward",
            "total_reward",
            "entrop",
        ],
        "plot_assemblage_tags": [
            ("total_reward",),
            ("entrop",),
        ],
        "use_normalized_rewards": False,
        "use_centered_reward": False,
        "use_rolling_avg_actor_grad": False,
        "process_reward_after_rolling": False,
        "only_process_reward": False,
        "use_rolling_avg_reward": False,
        "reward_processing_bais": 0.1,
        "center_and_normalize_with_rolling_avg": False,
        "punishment_helped": True,
        "gamma": 0.9,
        "lr": 0.01 if env == "ABCoinGame" else 0.005,
        "num_episodes": 3 if debug else 2000,
        "trace_length": 10 if debug else 40,
        "weigth_decay": 0.001875,
        "lola_correction_multiplier": 8,
        "entropy_coeff": 0.02,
        "batch_size": 12 if debug else 1024,
    }

    return tune_hparams


def _train(tune_hp):
    tune_config, stop, env_config = _get_tune_config(tune_hp)

    if "CoinGame" in tune_config["env_name"]:
        trainable_class = train_cg_tune_class_API.LOLAPGCG
    else:
        raise ValueError()

    # Train with the Tune Class API (not RLLib Class)
    experiment_analysis = tune.run(
        trainable_class,
        name=tune_hp["exp_name"],
        config=tune_config,
        checkpoint_at_end=True,
        stop=stop,
        metric=tune_config["metric"],
        mode="max",
        log_to_file=not tune_hp["debug"],
    )

    if tune_hp["remove_trials_below_speed"]:
        if not tune_hp["debug"]:
            experiment_analysis = _remove_failed_trials(
                experiment_analysis, tune_hp
            )

    if tune_hp["classify_into_welfare_fn"]:
        experiment_analysis_per_welfare = (
            _classify_trials_in_function_of_welfare(
                experiment_analysis, tune_hp
            )
        )
    else:
        experiment_analysis_per_welfare = {"": experiment_analysis}

    aggregate_and_plot_tensorboard_data.add_summary_plots(
        main_path=os.path.join("~/ray_results/", tune_config["exp_name"]),
        plot_keys=tune_config["plot_keys"],
        plot_assemble_tags_in_one_plot=tune_config["plot_assemblage_tags"],
    )

    return experiment_analysis_per_welfare


def _remove_failed_trials(results, tune_hp):
    if tune_hp["remove_trials_below_speed_for_both"]:
        results = exp_analysis.filter_trials_wt_n_metrics(
            results,
            metrics=("player_red_pick_speed", "player_blue_pick_speed"),
            metric_thresholds=(
                tune_hp["remove_trials_below_speed"],
                tune_hp["remove_trials_below_speed"],
            ),
            metric_modes=("last-5-avg", "last-5-avg"),
            threshold_modes=(exp_analysis.ABOVE, exp_analysis.ABOVE),
        )
    else:
        results = exp_analysis.filter_trials(
            results,
            metric="player_red_pick_speed",
            metric_threshold=tune_hp["remove_trials_below_speed"],
            metric_mode="last-5-avg",
            threshold_mode=exp_analysis.ABOVE,
        )
        results = exp_analysis.filter_trials(
            results,
            metric="player_blue_pick_speed",
            metric_threshold=tune_hp["remove_trials_below_speed"],
            metric_mode="last-5-avg",
            threshold_mode=exp_analysis.ABOVE,
        )
    return results


def _get_tune_config(tune_hp: dict, stop_on_epi_number: bool = False):
    tune_config = copy.deepcopy(tune_hp)

    assert not tune_config["exact"]

    # Resolve default parameters
    assert "CoinGame" in tune_config["env_name"]

    if tune_config["env_name"] == "CoinGame":
        tune_config["env_class"] = vectorized_coin_game.VectorizedCoinGame
    elif tune_config["env_name"] == "ABCoinGame":
        tune_config["env_class"] = vectorized_ab_coin_game.VectorizedABCG
    else:
        raise ValueError()

    tune_config["lr"] = (
        0.005 if tune_config["lr"] is None else tune_config["lr"]
    )
    tune_config["gamma"] = (
        0.96 if tune_config["gamma"] is None else tune_config["gamma"]
    )
    tune_hp["x_limits"] = (-0.1, 0.6)
    tune_hp["y_limits"] = (-0.1, 0.6)
    if tune_config["env_class"] == vectorized_coin_game.AsymVectorizedCoinGame:
        tune_hp["x_limits"] = (-1.0, 3.0)
    elif tune_config["env_class"] == vectorized_ab_coin_game.VectorizedABCG:
        tune_hp["x_limits"] = (-0.02, 0.8)
        tune_hp["y_limits"] = (-0.02, 1.5)

    tune_hp["jitter"] = 0.00
    env_config = {
        "players_ids": ["player_red", "player_blue"],
        "batch_size": tune.sample_from(lambda spec: spec.config["batch_size"]),
        "max_steps": tune.sample_from(
            lambda spec: spec.config["trace_length"]
        ),
        "grid_size": tune_config["grid_size"],
        "get_additional_info": True,
        "both_players_can_pick_the_same_coin": True,
        "force_vectorize": False,
        "same_obs_for_each_player": True,
        "punishment_helped": tune_config["punishment_helped"],
    }
    tune_config["metric"] = "player_blue_pick_speed"
    tune_config["plot_keys"] += (
        train_cg_tune_class_API.PLOT_KEYS
        + vectorized_coin_game.PLOT_KEYS
        + aggregate_and_plot_tensorboard_data.PLOT_KEYS
    )
    tune_config["plot_assemblage_tags"] += (
        train_cg_tune_class_API.PLOT_ASSEMBLAGE_TAGS
        + vectorized_coin_game.PLOT_ASSEMBLAGE_TAGS
        + aggregate_and_plot_tensorboard_data.PLOT_ASSEMBLAGE_TAGS
    )

    # For hyperparameter search
    tune_hp["scale_multipliers"] = tune.sample_from(
        lambda spec: (
            1 / spec.config["trace_length"],
            1 / spec.config["trace_length"],
        )
    )
    tune_config["env_config"] = env_config

    if stop_on_epi_number:
        stop = {"episodes_total": tune_config["num_episodes"]}
    else:
        stop = {"finished": True}

    return tune_config, stop, env_config


def _evaluate(tune_hp, debug, experiment_analysis_per_exp):
    (
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop,
        env_config,
    ) = _generate_eval_config(tune_hp, debug)

    _evaluate_self_and_cross_perf(
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop,
        env_config,
        experiment_analysis_per_exp,
    )


def _generate_eval_config(tune_hp, debug):
    rllib_hp = copy.deepcopy(tune_hp)
    rllib_hp["seed"] = 2020
    rllib_hp["num_episodes"] = 1 if debug else 100
    tune_config, stop, env_config = _get_tune_config(
        rllib_hp, stop_on_epi_number=True
    )
    rllib_hp["env_class"] = tune_config["env_class"]

    if "CoinGame" in tune_config["env_name"]:
        env_config["batch_size"] = 1
        tune_config["TuneTrainerClass"] = train_cg_tune_class_API.LOLAPGCG
    else:
        raise ValueError()
    tune_config["env_config"].update(
        {
            "batch_size": env_config["batch_size"],
            "max_steps": rllib_hp["trace_length"],
        }
    )
    rllib_hp["scale_multipliers"] = (
        1 / rllib_hp["trace_length"],
        1 / rllib_hp["trace_length"],
    )

    rllib_config_eval = {
        "env": rllib_hp["env_class"],
        "env_config": env_config,
        "multiagent": {
            "policies": {
                env_config["players_ids"][0]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env_class"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env_class"].ACTION_SPACE,
                    {"tune_config": tune_config},
                ),
                env_config["players_ids"][1]: (
                    policy.get_tune_policy_class(DQNTorchPolicy),
                    rllib_hp["env_class"](env_config).OBSERVATION_SPACE,
                    rllib_hp["env_class"].ACTION_SPACE,
                    {"tune_config": tune_config},
                ),
            },
            "policy_mapping_fn": lambda agent_id: agent_id,
            "policies_to_train": ["None"],
        },
        "seed": rllib_hp["seed"],
        "min_iter_time_s": 3.0,
        "callbacks": callbacks.merge_callbacks(
            log.get_logging_callbacks_class(
                log_full_epi=True,
            ),
            callbacks.PolicyCallbacks,
        ),
        "num_envs_per_worker": 1,
        "num_workers": 0,
    }

    policies_to_load = copy.deepcopy(env_config["players_ids"])

    if "CoinGame" in rllib_hp["env_name"]:
        trainable_class = train_cg_tune_class_API.LOLAPGCG
        rllib_config_eval["model"] = {
            "dim": env_config["grid_size"],
            # [Channel, [Kernel, Kernel], Stride]]
            "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1]],
        }
    else:
        raise ValueError()

    return (
        rllib_hp,
        rllib_config_eval,
        policies_to_load,
        trainable_class,
        stop,
        env_config,
    )


def _evaluate_self_and_cross_perf(
    rllib_hp,
    rllib_config_eval,
    policies_to_load,
    trainable_class,
    stop,
    env_config,
    experiment_analysis_per_welfare,
    n_cross_play_per_checkpoint=None,
):
    exp_name = os.path.join(rllib_hp["exp_name"], "eval")
    evaluator = cross_play.evaluator.SelfAndCrossPlayEvaluator(
        exp_name=exp_name,
        local_mode=rllib_hp["debug"],
    )
    analysis_metrics_per_mode = evaluator.perform_evaluation_or_load_data(
        evaluation_config=rllib_config_eval,
        stop_config=stop,
        policies_to_load_from_checkpoint=policies_to_load,
        experiment_analysis_per_welfare=experiment_analysis_per_welfare,
        tune_trainer_class=trainable_class,
        n_cross_play_per_checkpoint=min(5, rllib_hp["train_n_replicates"] - 1)
        if n_cross_play_per_checkpoint is None
        else n_cross_play_per_checkpoint,
        to_load_path=rllib_hp["load_plot_data"],
    )

    if issubclass(
        rllib_hp["env_class"],
        matrix_sequential_social_dilemma.MatrixSequentialSocialDilemma,
    ):
        background_area_coord = rllib_hp["env_class"].PAYOFF_MATRIX
    else:
        background_area_coord = None

    plot_config = PlotConfig(
        xlim=rllib_hp["x_limits"],
        ylim=rllib_hp["y_limits"],
        markersize=5,
        jitter=rllib_hp["jitter"],
        xlabel="player 1 payoffs",
        ylabel="player 2 payoffs",
        plot_max_n_points=rllib_hp["train_n_replicates"],
        x_scale_multiplier=rllib_hp["scale_multipliers"][0],
        y_scale_multiplier=rllib_hp["scale_multipliers"][1],
        background_area_coord=background_area_coord,
    )
    evaluator.plot_results(
        analysis_metrics_per_mode,
        plot_config=plot_config,
        x_axis_metric=f"policy_reward_mean/{env_config['players_ids'][0]}",
        y_axis_metric=f"policy_reward_mean/{env_config['players_ids'][1]}",
    )


def _classify_trials_in_function_of_welfare(experiment_analysis, hp):
    experiment_analysis_per_welfare = {}
    for trial in experiment_analysis.trials:
        welfare_name = _get_trial_welfare(trial, hp)
        if welfare_name not in experiment_analysis_per_welfare.keys():
            lola_exact_official._add_empty_experiment_analysis(
                experiment_analysis_per_welfare,
                welfare_name,
                experiment_analysis,
            )
        experiment_analysis_per_welfare[welfare_name].trials.append(trial)
    return experiment_analysis_per_welfare


def _get_trial_welfare(trial, hp):
    pick_own_player_1 = trial.last_result["player_red_pick_own_color"]
    pick_own_player_2 = trial.last_result["player_blue_pick_own_color"]
    reward_player_1 = trial.last_result["total_reward_player_red"]
    reward_player_2 = trial.last_result["total_reward_player_blue"]
    welfare_name = lola_pg_classify_fn(
        pick_own_player_1,
        pick_own_player_2,
        hp,
        reward_player_1,
        reward_player_2,
    )
    return welfare_name


FAILURE = "failures"
EGALITARIAN = "egalitarian"
UTILITARIAN = "utilitarian"


def lola_pg_classify_fn(
    pick_own_player_1, pick_own_player_2, hp, reward_player_1, reward_player_2
):
    if reward_player_2 != 0.0 and reward_player_1 != 0.0:
        if hp["env_name"] == "ABCoinGame":
            ratio = reward_player_2 / reward_player_1
        else:
            ratio = max(
                reward_player_1 / reward_player_2,
                reward_player_2 / reward_player_1,
            )
        if ratio > 1.2:
            return UTILITARIAN
    return EGALITARIAN


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
