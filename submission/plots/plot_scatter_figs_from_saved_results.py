import json
import os
from collections import Iterable

import matplotlib.pyplot as plt
import numpy as np

from submission.plots.plot_bar_chart_from_saved_results import (
    _get_inputs,
    Perf,
    Exp_data,
    Final_values,
    NA,
    PLAYER_0,
    PLAYER_1,
    COLORS,
    LEGEND_NO_SPLIT,
    LEGEND,
    REMOVE_STARS,
    VALUE_TERM,
)

plt.switch_backend("agg")
plt.style.use("seaborn-whitegrid")
plt.rcParams.update({"font.size": 12})

VECTICAL_PLOT = True
# PLOT_


def main(debug):
    prefix, files_data, n_players = _get_inputs()
    # files_data = list(files_data)
    # files_data = files_data[0:-4] + files_data[-2:]
    files_to_process = _preprocess_inputs(prefix, files_data)

    perf_per_mode_per_files = []
    for file_paths, file_data in zip(files_to_process, files_data):
        perf_per_mode = _get_stats(file_paths, n_players, file_data)
        perf_per_mode_per_files.append(
            Exp_data(file_data.base_algo, file_data.env, perf_per_mode)
        )

    _plot_ipd_iasymbos(perf_per_mode_per_files)
    _plot_all(perf_per_mode_per_files)
    _plot_ipd_iasymbos(perf_per_mode_per_files, welfare_split=False)
    _plot_all(perf_per_mode_per_files, welfare_split=False)


def _preprocess_inputs(prefix, files_data):

    # !!!! Remove Negotiation data (not plotted in scatter plot) !!!!
    files_data = files_data[:-1]

    files_to_process = []
    for file_data in files_data:
        if isinstance(file_data, Final_values):
            value = file_data
        elif file_data.path_to_preferences is not None:
            value = (
                os.path.join(prefix, file_data.path_to_self_play),
                os.path.join(prefix, file_data.path_to_preferences),
            )
        else:
            value = (
                os.path.join(prefix, file_data.path_to_self_play),
                None,
            )
        files_to_process.append(value)

    return files_to_process


def _get_stats(file_paths, n_players, file_data):
    if isinstance(file_paths, Final_values):
        all_perf = file_paths
    else:
        self_play_path = file_paths[0]
        perf_per_mode = _get_stats_for_file(
            self_play_path, n_players, file_data
        )
        self_play = perf_per_mode["self-play"]
        cross_play = perf_per_mode["cross-play"]

        preference_path = file_paths[1]
        if preference_path is not None:
            perf_per_mode_bis = _get_stats_for_file(
                preference_path, n_players, file_data
            )
            same_preferences_cross_play = perf_per_mode_bis[
                "cross-play: same pref vs same pref"
            ]
            if (
                "cross-play: diff pref vs diff pref"
                in perf_per_mode_bis.keys()
            ):
                diff_preferences_cross_play = perf_per_mode_bis[
                    "cross-play: diff pref vs diff pref"
                ]
            else:
                diff_preferences_cross_play = NA
        else:
            same_preferences_cross_play = NA
            diff_preferences_cross_play = NA

        all_perf = [
            self_play,
            cross_play,
            same_preferences_cross_play,
            diff_preferences_cross_play,
        ]
    return all_perf


def _get_stats_for_file(file, n_players, file_data):
    perf_per_mode = {}
    file_path = os.path.expanduser(file)
    with (open(file_path, "rb")) as f:
        file_content = json.load(f)
        for eval_mode, mode_perf in file_content.items():
            perf = [None] * 2
            print("eval_mode", eval_mode)
            for metric, metric_perf in mode_perf.items():
                player_idx = _extract_player_idx(metric)

                perf_per_replicat = np.array(
                    _convert_str_of_list_to_list(metric_perf["raw_data"])
                )

                n_replicates_in_content = len(perf_per_replicat)
                values_per_replicat_per_player = _scale_values(
                    perf_per_replicat, file_data
                )

                mean_per_player = values_per_replicat_per_player.mean(axis=0)
                std_dev_per_player = values_per_replicat_per_player.std(axis=0)
                std_err_per_player = std_dev_per_player / np.sqrt(
                    n_replicates_in_content
                )
                perf[player_idx] = Perf(
                    mean_per_player,
                    std_dev_per_player,
                    std_err_per_player,
                    values_per_replicat_per_player,
                )
            perf_per_mode[eval_mode] = perf

    return perf_per_mode


def _extract_player_idx(metric):
    if "player_row" in metric:
        player_idx = PLAYER_0
    elif "player_col" in metric:
        player_idx = PLAYER_1
    elif "player_red" in metric:
        player_idx = PLAYER_0
    elif "player_blue" in metric:
        player_idx = PLAYER_1
    else:
        raise ValueError()
    return player_idx


def _scale_values(values_per_replicat_per_player, file_data):
    scaled_values = (
        values_per_replicat_per_player / file_data.reward_adaptation_divider
    )
    return scaled_values


def _convert_str_of_list_to_list(str_of_list):
    return [
        float(v)
        for v in str_of_list.replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .split(",")
    ]


def _plot_all(perf_per_mode_per_files, welfare_split=True):
    n_figures = len(perf_per_mode_per_files)
    # n_row = int(np.sqrt(n_figures) + 0.99)
    # n_row = 4

    # n = 100 * n_row + 10 * n_row + 1
    if VECTICAL_PLOT:
        plt.figure(figsize=(6, 12))
        n = 421
    else:
        n = 241
        plt.figure(figsize=(10, 6))

    for i in range(n_figures):
        plt.subplot(n + i)
        data_idx = i
        if perf_per_mode_per_files[i].env == "IPD":
            xlim = (-3.5, 0.5)
            ylim = (-3.5, 0.5)
            jitter = 0.05
            background_area_coord = np.array(
                [[[-1, -1], [-3, +0]], [[+0, -3], [-2, -2]]]
            )
            _add_background_area(background_area_coord)
        elif perf_per_mode_per_files[i].env == "CG":
            xlim = (-0.1, 0.6)
            ylim = (-0.1, 0.6)
            jitter = 0.00
        elif perf_per_mode_per_files[i].env == "IAsymBoS":
            xlim = (-0.5, 4.5)
            ylim = (-0.5, 4.5)
            jitter = 0.05
            background_area_coord = np.array(
                [[[+4.0, +1.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +2.0]]]
            )
            _add_background_area(background_area_coord)
        elif perf_per_mode_per_files[i].env == "ABCG":
            xlim = (-0.1, 0.8)
            ylim = (-0.1, 1.6)
            jitter = 0.00
        if VECTICAL_PLOT:
            plot_x_label = True
            plot_y_label = True
        else:
            plot_x_label = i > 3
            plot_y_label = i % 4 == 0
        _plot_one_scatter(
            perf_per_mode_per_files,
            data_idx,
            xlim,
            ylim,
            jitter,
            welfare_split=welfare_split,
            plot_x_label=plot_x_label,  # i // n_row == n_row - 1 or i == 5,
            plot_y_label=plot_y_label,
        )

    if VECTICAL_PLOT:
        plt.tight_layout(rect=[0, 0.08, 1.0, 1.0])
    else:
        plt.tight_layout(rect=[0, 0.13, 1.0, 1.0])
    # if welfare_split:
    #     # Save the figure and show
    #     plt.legend(
    #         LEGEND if welfare_split else LEGEND_NO_SPLIT,
    #         frameon=True,
    #         bbox_to_anchor=(1.0, -0.25),
    #     )
    #
    # else:
    #     # plt.tight_layout(rect=[0, 0.13, 1.0, 1.0])
    #     # Save the figure and show
    #     plt.legend(
    #         LEGEND if welfare_split else LEGEND_NO_SPLIT,
    #         frameon=True,
    #         # bbox_to_anchor=(2.0, 0.55),
    #         bbox_to_anchor=(1.0, -0.25),
    #     )
    if VECTICAL_PLOT:
        plt.legend(
            LEGEND if welfare_split else LEGEND_NO_SPLIT,
            frameon=True,
            bbox_to_anchor=(0.6, -0.30) if welfare_split else (0.15, -0.30),
        )
    else:
        plt.legend(
            LEGEND if welfare_split else LEGEND_NO_SPLIT,
            frameon=True,
            bbox_to_anchor=(1.0, -0.25),
        )
    plt.savefig(f"scatter_plots_all_split_{welfare_split}.png")


def _plot_ipd_iasymbos(perf_per_mode_per_files, welfare_split=True):
    if welfare_split:
        plt.figure(figsize=(5.8, 5 * 2 / 3))
    else:
        plt.figure(figsize=(5, 2.5))

    margin = 0.1

    from matplotlib import gridspec

    if welfare_split:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    else:
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])
    # ax0 = plt.subplot(gs[0])

    # plt.subplot(121)
    plt.subplot(gs[0])
    # , gridspec_kw = {"width_ratios": [1, 2]}
    data_idx = 1
    xlim = (-3.0 - margin, 0.0 + margin)
    if welfare_split:
        ylim = (-3 - 3.0 - margin, +3 + 0.0 + margin)
    else:
        ylim = (-3.0 - margin, 0.0 + margin)
    jitter = 0.05
    _plot_one_scatter(
        perf_per_mode_per_files,
        data_idx,
        xlim,
        ylim,
        jitter,
        welfare_split=welfare_split,
    )
    background_area_coord = np.array(
        [[[-1, -1], [-3, +0]], [[+0, -3], [-2, -2]]]
    )
    _add_background_area(background_area_coord)

    # plt.subplot(122)
    plt.subplot(gs[1])
    data_idx = 5
    if welfare_split:
        xlim = (-0.0 - margin * 2, 4.0 + margin * 2)
        ylim = (-0.0 - margin, 3.5 + margin)
    else:
        xlim = (-0.0 - margin * 2, 4.0 + margin * 2)
        ylim = (-0.0 - margin, 2.0 + margin)
    jitter = 0.05
    _plot_one_scatter(
        perf_per_mode_per_files,
        data_idx,
        xlim,
        ylim,
        jitter,
        welfare_split=welfare_split,
        plot_y_label=False,
    )
    background_area_coord = np.array(
        [[[+4.0, +1.0], [+0.0, +0.0]], [[+0.0, +0.0], [+2.0, +2.0]]]
    )
    _add_background_area(background_area_coord)

    # plt.text(
    #     -1.0,
    #     -1.50,
    #     "c)",
    #     fontdict={"fontsize": 14.0, "weight": "bold"},
    # )

    plt.tight_layout(rect=[0, 0.0, 1.0, 1.0])
    if welfare_split:
        plt.legend(
            LEGEND if welfare_split else LEGEND_NO_SPLIT,
            frameon=True,
            bbox_to_anchor=(0.01, 0.0, 1.0, 1.0),
        )
    else:
        plt.legend(
            LEGEND if welfare_split else LEGEND_NO_SPLIT,
            frameon=True,
            bbox_to_anchor=(-0.185, 0.46),
        )
    # plt.tight_layout(rect=[0, -0.07, 1.0, 1.0])

    # Save the figure and show
    plt.savefig(f"scatter_plot_ipd_iasymbos_split_{welfare_split}.png")


def _plot_one_scatter(
    perf_per_mode_per_files,
    data_idx,
    xlim,
    ylim,
    jitter,
    plot_x_label=True,
    plot_y_label=True,
    welfare_split=True,
):
    _plot(perf_per_mode_per_files, data_idx, jitter, welfare_split)
    env = perf_per_mode_per_files[data_idx].env
    base_algo = perf_per_mode_per_files[data_idx].base_algo
    if REMOVE_STARS:
        env = env.replace("*", "").strip()
        base_algo = base_algo.replace("*", "").strip()
    plt.title(f"{env} + " f"{base_algo}")
    if plot_x_label:
        plt.xlabel(f"Player 1 {VALUE_TERM}")
    if plot_y_label:
        plt.ylabel(f"Player 2 {VALUE_TERM}")
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)


def _plot(perf_per_mode_per_files, data_idx, jitter, welfare_split):
    all_perf = [el.perf for el in perf_per_mode_per_files]

    (
        self_play_p0,
        self_play_p1,
        cross_play_p0,
        cross_play_p1,
        same_pref_p0,
        same_pref_p1,
        diff_pref_p0,
        diff_pref_p1,
    ) = _preproces_values(all_perf, jitter)

    self_play_p0 = self_play_p0[data_idx]
    self_play_p1 = self_play_p1[data_idx]
    cross_play_p0 = cross_play_p0[data_idx]
    cross_play_p1 = cross_play_p1[data_idx]
    same_pref_p0 = same_pref_p0[data_idx]
    same_pref_p1 = same_pref_p1[data_idx]
    diff_pref_p0 = diff_pref_p0[data_idx]
    diff_pref_p1 = diff_pref_p1[data_idx]

    print("cross_play_p0", len(cross_play_p0))
    print("np.array(same_pref_p0).shape", np.array(same_pref_p0).shape)
    plot_diff_pref = True
    if np.array(same_pref_p0).shape == ():
        same_pref_p0 = cross_play_p0
        same_pref_p1 = cross_play_p1
        plot_diff_pref = False

    (
        self_play_p0,
        self_play_p1,
        cross_play_p0,
        cross_play_p1,
        same_pref_p0,
        same_pref_p1,
        diff_pref_p0,
        diff_pref_p1,
    ) = _keep_same_number_of_points(
        self_play_p0,
        self_play_p1,
        cross_play_p0,
        cross_play_p1,
        same_pref_p0,
        same_pref_p1,
        diff_pref_p0,
        diff_pref_p1,
    )

    plt.plot(
        self_play_p0,
        self_play_p1,
        markerfacecolor="none",
        markeredgecolor=COLORS[0],
        linestyle="None",
        marker="o",
        color=COLORS[0],
        # markersize=MARKERSIZE,
    )
    if welfare_split:
        plt.plot(
            same_pref_p0,
            same_pref_p1,
            markerfacecolor="none",
            markeredgecolor=COLORS[1],
            linestyle="None",
            marker="s",
            color=COLORS[1],
            # markersize=self.plot_cfg.markersize,
        )
        if plot_diff_pref:
            plt.plot(
                diff_pref_p0,
                diff_pref_p1,
                markerfacecolor="none",
                markeredgecolor=COLORS[2],
                linestyle="None",
                marker="v",
                color=COLORS[2],
                # markersize=self.plot_cfg.markersize,
            )
    else:
        plt.plot(
            cross_play_p0,
            cross_play_p1,
            markerfacecolor="none",
            markeredgecolor=COLORS[1],
            linestyle="None",
            marker="s",
            color=COLORS[1],
            # markersize=self.plot_cfg.markersize,
        )


def _keep_same_number_of_points(*args):
    lengths = [len(list_) for list_ in args if isinstance(list_, Iterable)]
    min_length = min(lengths)
    new_lists = []
    for list_ in args:
        if isinstance(list_, Iterable):
            list_ = list_[-min_length:]
        new_lists.append(list_)

    return new_lists


def _add_jitter(values, jitter):
    values_wt_jitter = []
    for sub_list in values:
        sub_list = np.array(sub_list)
        shift = np.random.normal(0.0, jitter, sub_list.shape)
        sub_list += shift
        values_wt_jitter.append(sub_list.tolist())
    return values_wt_jitter


def _preproces_values(all_perf, jitter):
    self_play_p0 = _extract_value(all_perf, 0, PLAYER_0, "raw", jitter)
    self_play_p1 = _extract_value(all_perf, 0, PLAYER_1, "raw", jitter)
    cross_play_p0 = _extract_value(all_perf, 1, PLAYER_0, "raw", jitter)
    cross_play_p1 = _extract_value(all_perf, 1, PLAYER_1, "raw", jitter)
    same_pref_p0 = _extract_value(all_perf, 2, PLAYER_0, "raw", jitter)
    same_pref_p1 = _extract_value(all_perf, 2, PLAYER_1, "raw", jitter)
    diff_pref_p0 = _extract_value(all_perf, 3, PLAYER_0, "raw", jitter)
    diff_pref_p1 = _extract_value(all_perf, 3, PLAYER_1, "raw", jitter)

    return (
        self_play_p0,
        self_play_p1,
        cross_play_p0,
        cross_play_p1,
        same_pref_p0,
        same_pref_p1,
        diff_pref_p0,
        diff_pref_p1,
    )


def _log_n_replicates(
    self_play,
    cross_play,
    same_pref_perf,
    diff_pref_perf,
):
    print("\n_log_n_replicates")
    print("self_play", [el.shape for el in self_play])
    print("cross_play", [el.shape for el in cross_play])
    print("same_pref_perf", [el.shape for el in same_pref_perf])
    print("diff_pref_perf", [el.shape for el in diff_pref_perf])

    ratio = []
    for cross, cross_same, cross_diff in zip(
        cross_play, same_pref_perf, diff_pref_perf
    ):
        if len(cross_same.shape) > 0:
            assert cross.shape[0] == (
                cross_same.shape[0] + cross_diff.shape[0]
            )
            ratio.append(cross_same.shape[0] / cross_diff.shape[0])
        else:
            ratio.append(None)
    print("cross_same / cross_diff", ratio)


def _extract_value(all_perf, idx, player_idx, attrib, jitter):
    values = []
    for el in all_perf:
        if isinstance(el, Final_values):
            values.append(0.0)
        else:
            if hasattr(el[idx][player_idx], attrib):
                values.append(getattr(el[idx][player_idx], attrib))
            else:
                values.append(0.0)

    values = _add_jitter(values, jitter)

    return values


def _add_background_area(background_area_coord):
    from scipy.spatial import ConvexHull

    assert background_area_coord.ndim == 3
    points_defining_area = background_area_coord.flatten().reshape(-1, 2)
    area_hull = ConvexHull(points_defining_area)
    plt.fill(
        points_defining_area[area_hull.vertices, 0],
        points_defining_area[area_hull.vertices, 1],
        facecolor="none",
        edgecolor="purple",
        linewidth=1,
    )
    plt.fill(
        points_defining_area[area_hull.vertices, 0],
        points_defining_area[area_hull.vertices, 1],
        "purple",
        alpha=0.05,
    )


if __name__ == "__main__":
    debug_mode = False
    main(debug_mode)
