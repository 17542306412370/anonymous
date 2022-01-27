import random

import numpy as np

from submission.envs.ab_coin_game import ABCoinGame
from coin_game_tests_utils import (
    check_custom_obs,
    assert_logger_buffer_size,
    helper_test_reset,
    helper_test_step,
    init_several_envs,
    helper_test_multiple_steps,
    helper_test_multi_ple_episodes,
    helper_assert_info,
    get_random_action,
)


# TODO add tests for grid_size != 3


def init_my_envs(
    max_steps,
    grid_size,
    players_can_pick_same_coin=True,
    same_obs_for_each_player=True,
    use_other_play=False,
):
    return init_several_envs(
        (ABCoinGame,),
        max_steps=max_steps,
        grid_size=grid_size,
        players_can_pick_same_coin=players_can_pick_same_coin,
        same_obs_for_each_player=same_obs_for_each_player,
        use_other_play=use_other_play,
    )


def check_obs(obs, grid_size):
    check_custom_obs(obs, grid_size, n_in_2_and_above=2.0, n_layers=6)


def test_reset():
    max_steps, grid_size = 20, 3
    envs = init_my_envs(max_steps, grid_size)
    helper_test_reset(envs, check_obs, grid_size=grid_size)


def test_step():
    max_steps, grid_size = 20, 3
    envs = init_my_envs(max_steps, grid_size)
    helper_test_step(envs, check_obs, grid_size=grid_size)


def test_multiple_steps():
    max_steps, grid_size = 20, 3
    n_steps = int(max_steps * 0.75)
    envs = init_my_envs(max_steps, grid_size)
    helper_test_multiple_steps(
        envs,
        n_steps,
        check_obs,
        grid_size=grid_size,
    )


def test_multiple_episodes():
    max_steps, grid_size = 20, 3
    n_steps = int(max_steps * 8.25)
    envs = init_my_envs(max_steps, grid_size)
    helper_test_multi_ple_episodes(
        envs,
        n_steps,
        max_steps,
        check_obs,
        grid_size=grid_size,
    )


def overwrite_pos(
    env,
    p_red_pos,
    p_blue_pos,
    c_red_pos,
    c_blue_pos,
    **kwargs,
):
    env.red_coin = True
    env.red_pos = p_red_pos
    env.blue_pos = p_blue_pos
    env.red_coin_pos = c_red_pos
    env.blue_coin_pos = c_blue_pos

    env.red_pos = np.array(env.red_pos)
    env.blue_pos = np.array(env.blue_pos)
    env.red_coin_pos = np.array(env.red_coin_pos)
    env.blue_coin_pos = np.array(env.blue_coin_pos)


def assert_not_present_in_dict_or_equal(key, value, info, player):
    if value is None:
        assert key not in info[player]
    else:
        assert info[player][key] == value


def test_logged_info_no_picking():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=None,
    )


def test_logged_info__red_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=None,
    )


def test_logged_info__blue_pick_red_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=None,
    )


def test_logged_info__blue_pick_blue_all_the_time():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=1.0,
        red_own=None,
        blue_own=1.0,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=0.0,
    )


def test_logged_info__red_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=None,
    )


def test_logged_info__both_pick_blue_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=1.0,
        red_own=None,
        blue_own=1.0,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=0.0,
    )


def test_logged_info__both_pick_red_all_the_time():
    p_red_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [1, 0], [1, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=1.0,
        blue_speed=1.0,
        red_own=1.0,
        blue_own=0.0,
        red_coop_speed=1.0,
        blue_coop_speed=0.0,
        red_coop_fraction=1.0,
        blue_coop_fraction=1.0,
    )


def test_logged_info__both_pick_red_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    c_blue_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.0,
        red_own=None,
        blue_own=None,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=None,
    )


def test_logged_info__both_pick_blue_half_the_time():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.5,
        red_own=None,
        blue_own=1.0,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=0.0,
    )


def test_logged_info__both_pick_blue():
    p_red_pos = [[0, 0], [0, 0], [0, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[0, 0], [0, 0], [0, 0], [0, 0]]
    c_blue_pos = [[1, 1], [1, 1], [1, 1], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.5,
        red_own=None,
        blue_own=1.0,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=0.0,
    )


def test_logged_info__pick_half_the_time_half_blue_half_red():
    p_red_pos = [[0, 0], [0, 0], [1, 0], [1, 0]]
    p_blue_pos = [[1, 0], [1, 0], [0, 0], [0, 0]]
    p_red_act = [0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0]
    c_red_pos = [[1, 1], [0, 0], [1, 1], [0, 0]]
    c_blue_pos = [[0, 0], [1, 1], [0, 0], [1, 1]]
    max_steps, grid_size = 4, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size)

    helper_assert_info(
        n_steps=n_steps,
        p_red_act=p_red_act,
        p_blue_act=p_blue_act,
        envs=envs,
        grid_size=grid_size,
        max_steps=max_steps,
        p_red_pos=p_red_pos,
        p_blue_pos=p_blue_pos,
        c_red_pos=c_red_pos,
        c_blue_pos=c_blue_pos,
        check_obs_fn=check_obs,
        overwrite_pos_fn=overwrite_pos,
        red_speed=0.0,
        blue_speed=0.25,
        red_own=None,
        blue_own=1.0,
        red_coop_speed=0.0,
        blue_coop_speed=0.0,
        red_coop_fraction=None,
        blue_coop_fraction=0.0,
    )


def test_observations_are_not_invariant_to_the_player_trained_in_reset():
    p_red_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [2, 0],
        [0, 1],
        [2, 2],
        [1, 2],
    ]
    p_blue_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 0],
        [1, 2],
        [2, 2],
    ]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [
        [1, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
    ]
    c_blue_pos = [
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
        [0, 0],
    ]
    max_steps, grid_size = 10, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, same_obs_for_each_player=True)

    for env_i, env in enumerate(envs):
        obs = env.reset()
        assert_obs_is_not_symmetrical(obs, env)
        step_i = 0
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
        )

        for _ in range(n_steps):
            step_i += 1
            actions = {
                "player_red": p_red_act[step_i - 1],
                "player_blue": p_blue_act[step_i - 1],
            }
            _, _, _, _ = env.step(actions)

            if step_i == max_steps:
                break

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                c_red_pos[step_i],
                c_blue_pos[step_i],
            )


def assert_obs_is_not_symmetrical(obs, env):
    assert np.all(obs[env.players_ids[0]] == obs[env.players_ids[1]])


def test_observations_are_not_invariant_to_the_player_trained_in_step():
    p_red_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [0, 0],
        [1, 1],
        [2, 0],
        [0, 1],
        [2, 2],
        [1, 2],
    ]
    p_blue_pos = [
        [0, 0],
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 0],
        [1, 2],
        [2, 2],
    ]
    p_red_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    p_blue_act = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c_red_pos = [
        [1, 1],
        [0, 0],
        [0, 1],
        [0, 0],
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
    ]
    c_blue_pos = [
        [0, 0],
        [1, 1],
        [0, 0],
        [0, 1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0],
        [2, 1],
        [0, 0],
    ]
    max_steps, grid_size = 10, 3
    n_steps = max_steps
    envs = init_my_envs(max_steps, grid_size, same_obs_for_each_player=True)

    for env_i, env in enumerate(envs):
        _ = env.reset()
        step_i = 0
        overwrite_pos(
            env,
            p_red_pos[step_i],
            p_blue_pos[step_i],
            c_red_pos[step_i],
            c_blue_pos[step_i],
        )

        for _ in range(n_steps):
            step_i += 1
            actions = {
                "player_red": p_red_act[step_i - 1],
                "player_blue": p_blue_act[step_i - 1],
            }
            obs, reward, done, info = env.step(actions)

            # assert observations are symmetrical respective to the actions
            if step_i % 2 == 1:
                obs_step_odd = obs
            elif step_i % 2 == 0:
                assert np.any(
                    obs[env.players_ids[0]] != obs_step_odd[env.players_ids[1]]
                )
                assert np.any(
                    obs[env.players_ids[1]] != obs_step_odd[env.players_ids[0]]
                )
            assert_obs_is_not_symmetrical(obs, env)

            if step_i == max_steps:
                break

            overwrite_pos(
                env,
                p_red_pos[step_i],
                p_blue_pos[step_i],
                c_red_pos[step_i],
                c_blue_pos[step_i],
            )


def test_other_play_reset():
    for _ in range(10):
        max_steps, grid_size = 20, 3
        envs = init_my_envs(max_steps, grid_size, use_other_play=True)
        for env in envs:
            obs_wt_other_play = env.reset()
            obs_wtout_other_play = _generate_vanilla_obs(env)
            _check_against_manual_symmetries(
                env, obs_wtout_other_play, obs_wt_other_play
            )


def _generate_vanilla_obs(env):
    obs = env._generate_observation()
    obs_wtout_other_play = {
        env.player_red_id: obs[0],
        env.player_blue_id: obs[1],
    }
    return obs_wtout_other_play


def _check_against_manual_symmetries(
    env, obs_wtout_other_play, obs_wt_other_play
):

    for player_i in range(env.NUM_AGENTS):
        pl_obs_wtout_other_play = obs_wtout_other_play[
            env.players_ids[player_i]
        ]
        pl_obs_wt_other_play = obs_wt_other_play[env.players_ids[player_i]]
        pl_obs_wtout_other_play = reverse_symmetry(
            env, player_i, pl_obs_wtout_other_play
        )
        assert np.all(pl_obs_wtout_other_play == pl_obs_wt_other_play)


def reverse_symmetry(env, player_i, obs):

    if env._symmetries_in_use[player_i]["name"] == "identity":
        pass
    elif env._symmetries_in_use[player_i]["name"] == "flip_h":
        obs = np.flip(obs, axis=[-2])
    elif env._symmetries_in_use[player_i]["name"] == "flip_v":
        obs = np.flip(obs, axis=[-3])
    elif env._symmetries_in_use[player_i]["name"] == "flip_hv":
        obs = np.flip(obs, axis=[-3, -2])
    else:
        raise NotImplementedError()
    return obs


def test_other_play_step():
    for _ in range(10):
        max_steps, grid_size = 20, 3
        envs = init_my_envs(max_steps, grid_size, use_other_play=True)
        for env in envs:
            _ = env.reset()
            symmetries_names = [sym["name"] for sym in env._symmetries_in_use]
            obs_wtout_other_play_after_reset = _generate_vanilla_obs(env)

            actions = get_random_action(env)
            obs_wt_other_play, reward, done, info = env.step(actions)

            _assert_symmetries_stable_during_episode(env, symmetries_names)

            obs_wtout_other_play = _generate_vanilla_obs(env)
            _check_against_manual_symmetries(
                env, obs_wtout_other_play, obs_wt_other_play
            )
            for player_i in range(env.NUM_AGENTS):
                print("player_i", player_i)
                print("actions", actions)
                _check_symmetries_on_actions_are_working(
                    env,
                    actions,
                    player_i,
                    obs_wtout_other_play_after_reset,
                    obs_wt_other_play,
                    obs_wtout_other_play,
                )


def _check_symmetries_on_actions_are_working(
    env,
    actions,
    player_i,
    obs_wtout_other_play_after_reset,
    obs_wt_other_play,
    obs_wtout_other_play,
):
    obs_init_pli, obs_after_step_wtout_sym = _simulate_move_wtout_sym(
        env, player_i, actions, obs_wtout_other_play_after_reset
    )

    pl_reversed_obs_wt_other_play = reverse_symmetry(
        env, player_i, obs_wt_other_play[env.players_ids[player_i]]
    )[..., player_i]

    print("obs_init_pli", obs_init_pli)
    print("obs_after_step_wtout_sym", obs_after_step_wtout_sym)
    print(
        "obs_wtout_other_play",
        obs_wtout_other_play[env.players_ids[player_i]][..., player_i],
    )
    print(
        "pl_reversed_obs_wt_other_play",
        pl_reversed_obs_wt_other_play,
    )
    assert np.all(obs_after_step_wtout_sym == pl_reversed_obs_wt_other_play)


def _assert_symmetries_stable_during_episode(env, symmetries_names):
    assert all(
        [
            sym["name"] == symmetry_name
            for sym, symmetry_name in zip(
                env._symmetries_in_use, symmetries_names
            )
        ]
    )


def _simulate_move_wtout_sym(
    env, player_i, actions, obs_wtout_other_play_after_reset
):
    obs_init_pli = obs_wtout_other_play_after_reset[env.players_ids[player_i]][
        ..., player_i
    ]
    action_in_sym_space = actions[env.players_ids[player_i]]
    action_in_vanilla_space = env._symmetries_in_use[player_i]["act_sym"](
        action_in_sym_space
    )

    player_pos = np.nonzero(obs_init_pli)
    new_player_pos = [
        (pos + move) % env.grid_size
        for pos, move in zip(
            player_pos,
            env.MOVES[action_in_vanilla_space],
        )
    ]
    obs_after_step_wtout_sym = np.zeros_like(obs_init_pli)
    obs_after_step_wtout_sym[new_player_pos] = 1
    return obs_init_pli, obs_after_step_wtout_sym
