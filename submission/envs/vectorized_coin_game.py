import copy
from collections import Iterable

import numpy as np
from numba import jit, prange
from numba.typed import List
from ray.rllib.utils import override

from submission.envs import coin_game

PLOT_KEYS = coin_game.PLOT_KEYS
PLOT_ASSEMBLAGE_TAGS = coin_game.PLOT_ASSEMBLAGE_TAGS


class VectorizedCoinGame(coin_game.CoinGame):
    """
    Vectorized Coin Game environment.
    """

    def __init__(self, config={}):

        super().__init__(config)
        assert self.NUM_ACTIONS == 4

        self.batch_size = config.get("batch_size", 1)
        self.force_vectorized = config.get("force_vectorize", False)
        self.punishment_helped = config.get("punishment_helped", False)
        assert (
            self.grid_size == 3
        ), "hardcoded in the generate_state numba function"

    @override(coin_game.CoinGame)
    def _randomize_color_and_player_positions(self):
        # Reset coin color and the players and coin positions
        self.red_coin = np.random.randint(low=0, high=2, size=self.batch_size)
        self.red_pos = np.random.randint(
            self.grid_size, size=(self.batch_size, 2)
        )
        self.blue_pos = np.random.randint(
            self.grid_size, size=(self.batch_size, 2)
        )
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)

        self._players_do_not_overlap_at_start()

    @override(coin_game.CoinGame)
    def _players_do_not_overlap_at_start(self):
        for i in range(self.batch_size):
            while _same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = np.random.randint(self.grid_size, size=2)

    @override(coin_game.CoinGame)
    def _generate_coin(self):
        generate = np.ones(self.batch_size, dtype=bool)
        self.coin_pos, self.red_coin = generate_coin_wt_numba_optimization(
            self.batch_size,
            generate,
            self.red_coin,
            self.red_pos,
            self.blue_pos,
            self.coin_pos,
            self.grid_size,
        )

    @override(coin_game.CoinGame)
    def _generate_observation(self):
        obs = generate_observations_wt_numba_optimization(
            self.batch_size,
            self.red_pos,
            self.blue_pos,
            self.coin_pos,
            self.red_coin,
            self.grid_size,
        )

        obs = self._apply_optional_invariance_to_the_player_trained(obs)
        obs, _ = self._optional_unvectorize(obs)
        # print("env nonzero obs", np.nonzero(obs[0]))
        return obs

    def _optional_unvectorize(self, obs, rewards=None):
        if self.batch_size == 1 and not self.force_vectorized:
            obs = [one_obs[0, ...] for one_obs in obs]
            if rewards is not None:
                rewards[0], rewards[1] = rewards[0][0], rewards[1][0]
        return obs, rewards

    @override(coin_game.CoinGame)
    def step(self, actions: Iterable):
        # print("step")
        # print("env self.red_coin", self.red_coin)
        # print("env self.red_pos", self.red_pos)
        # print("env self.blue_pos", self.blue_pos)
        # print("env self.coin_pos", self.coin_pos)
        actions = self._from_RLLib_API_to_list(actions)
        self.step_count_in_current_episode += 1

        (
            self.red_pos,
            self.blue_pos,
            rewards,
            self.coin_pos,
            observation,
            self.red_coin,
            red_pick_any,
            red_pick_red,
            blue_pick_any,
            blue_pick_blue,
        ) = vectorized_step_wt_numba_optimization(
            actions,
            self.batch_size,
            self.red_pos,
            self.blue_pos,
            self.coin_pos,
            self.red_coin,
            self.grid_size,
            self.asymmetric,
            self.max_steps,
            self.both_players_can_pick_the_same_coin,
            self.punishment_helped,
        )

        if self.output_additional_info:
            self._accumulate_info(
                red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue
            )

        obs = self._apply_optional_invariance_to_the_player_trained(
            observation
        )
        obs, rewards = self._optional_unvectorize(obs, rewards)
        # print("env actions", actions)
        # print("env rewards", rewards)
        # print("env self.red_coin", self.red_coin)
        # print("env self.red_pos", self.red_pos)
        # print("env self.blue_pos", self.blue_pos)
        # print("env self.coin_pos", self.coin_pos)
        # print("env nonzero obs", np.nonzero(obs[0]))
        return self._to_RLLib_API(obs, rewards)

    @override(coin_game.CoinGame)
    def _get_episode_info(self, n_steps_played=None):
        n_steps_played = (
            n_steps_played
            if n_steps_played is not None
            else len(self.red_pick) * self.batch_size
        )
        return super()._get_episode_info(n_steps_played=n_steps_played)

    @override(coin_game.CoinGame)
    def _from_RLLib_API_to_list(self, actions):

        ac_red = actions[self.player_red_id]
        ac_blue = actions[self.player_blue_id]
        if not isinstance(ac_red, Iterable):
            assert not isinstance(ac_blue, Iterable)
            ac_red, ac_blue = [ac_red], [ac_blue]
        actions = [ac_red, ac_blue]
        actions = np.array(actions).T
        return actions

    def _save_env(self):
        env_save_state = {
            "red_pos": self.red_pos,
            "blue_pos": self.blue_pos,
            "coin_pos": self.coin_pos,
            "red_coin": self.red_coin,
            "grid_size": self.grid_size,
            "asymmetric": self.asymmetric,
            "batch_size": self.batch_size,
            "step_count_in_current_episode": self.step_count_in_current_episode,
            "max_steps": self.max_steps,
            "red_pick": self.red_pick,
            "red_pick_own": self.red_pick_own,
            "blue_pick": self.blue_pick,
            "blue_pick_own": self.blue_pick_own,
            "both_players_can_pick_the_same_coin": self.both_players_can_pick_the_same_coin,
        }
        return copy.deepcopy(env_save_state)

    def _load_env(self, env_state):
        for k, v in env_state.items():
            self.__setattr__(k, v)


class AsymVectorizedCoinGame(VectorizedCoinGame):
    NAME = "AsymCoinGame"

    def __init__(self, config={}):
        if "asymmetric" in config:
            assert config["asymmetric"]
        else:
            config["asymmetric"] = True
        super().__init__(config)


@jit(nopython=True)
def vectorized_step_wt_numba_optimization(
    actions,
    batch_size,
    red_pos,
    blue_pos,
    coin_pos,
    red_coin,
    grid_size: int,
    asymmetric: bool,
    max_steps: int,
    both_players_can_pick_the_same_coin: bool,
    punishment_helped: bool,
):
    red_pos, blue_pos = move_players(
        batch_size, actions, red_pos, blue_pos, grid_size
    )

    (
        reward,
        generate,
        red_pick_any,
        red_pick_red,
        blue_pick_any,
        blue_pick_blue,
    ) = compute_reward(
        batch_size,
        red_pos,
        blue_pos,
        coin_pos,
        red_coin,
        asymmetric,
        both_players_can_pick_the_same_coin,
        punishment_helped,
    )

    coin_pos, red_coin = generate_coin_wt_numba_optimization(
        batch_size, generate, red_coin, red_pos, blue_pos, coin_pos, grid_size
    )

    obs = generate_observations_wt_numba_optimization(
        batch_size, red_pos, blue_pos, coin_pos, red_coin, grid_size
    )

    return (
        red_pos,
        blue_pos,
        reward,
        coin_pos,
        obs,
        red_coin,
        red_pick_any,
        red_pick_red,
        blue_pick_any,
        blue_pick_blue,
    )


@jit(nopython=True)
def move_players(batch_size, actions, red_pos, blue_pos, grid_size):
    moves = List(
        [
            np.array([0, 1]),
            np.array([0, -1]),
            np.array([1, 0]),
            np.array([-1, 0]),
        ]
    )

    for j in prange(batch_size):
        red_pos[j] = (red_pos[j] + moves[actions[j, 0]]) % grid_size
        blue_pos[j] = (blue_pos[j] + moves[actions[j, 1]]) % grid_size
    return red_pos, blue_pos


@jit(nopython=True)
def compute_reward(
    batch_size,
    red_pos,
    blue_pos,
    coin_pos,
    red_coin,
    asymmetric,
    both_players_can_pick_the_same_coin,
    punishment_helped,
):
    reward_red = np.zeros(batch_size)
    reward_blue = np.zeros(batch_size)
    generate = np.zeros(batch_size, dtype=np.bool_)
    red_pick_any, red_pick_red, blue_pick_any, blue_pick_blue = 0, 0, 0, 0

    for i in prange(batch_size):
        red_first_if_both = None
        if not both_players_can_pick_the_same_coin:
            if _same_pos(red_pos[i], coin_pos[i]) and _same_pos(
                blue_pos[i], coin_pos[i]
            ):
                red_first_if_both = bool(np.random.randint(low=0, high=2))

        if red_coin[i]:
            if _same_pos(red_pos[i], coin_pos[i]) and (
                red_first_if_both is None or red_first_if_both
            ):
                generate[i] = True
                reward_red[i] += 1
                if asymmetric:
                    reward_red[i] += 3
                red_pick_any += 1
                red_pick_red += 1
            if _same_pos(blue_pos[i], coin_pos[i]) and (
                red_first_if_both is None or not red_first_if_both
            ):
                generate[i] = True
                reward_red[i] += -2
                reward_blue[i] += 1
                blue_pick_any += 1
                if asymmetric and punishment_helped:
                    reward_red[i] -= 6
        else:
            if _same_pos(red_pos[i], coin_pos[i]) and (
                red_first_if_both is None or red_first_if_both
            ):
                generate[i] = True
                reward_red[i] += 1
                reward_blue[i] += -2
                if asymmetric:
                    reward_red[i] += 3
                red_pick_any += 1
            if _same_pos(blue_pos[i], coin_pos[i]) and (
                red_first_if_both is None or not red_first_if_both
            ):
                generate[i] = True
                reward_blue[i] += 1
                blue_pick_any += 1
                blue_pick_blue += 1
    reward = [reward_red, reward_blue]

    return (
        reward,
        generate,
        red_pick_any,
        red_pick_red,
        blue_pick_any,
        blue_pick_blue,
    )


@jit(nopython=True)
def _same_pos(x, y):
    return (x == y).all()


@jit(nopython=True)
def generate_coin_wt_numba_optimization(
    batch_size, generate, red_coin, red_pos, blue_pos, coin_pos, grid_size
):
    red_coin[generate] = 1 - red_coin[generate]
    for i in prange(batch_size):
        if generate[i]:
            coin_pos[i] = _place_coin(red_pos[i], blue_pos[i], grid_size)
    return coin_pos, red_coin


@jit(nopython=True)
def _place_coin(red_pos_i, blue_pos_i, grid_size):
    red_pos_flat = _flatten_index(red_pos_i, grid_size)
    blue_pos_flat = _flatten_index(blue_pos_i, grid_size)
    possible_coin_pos = np.array(
        [x for x in range(9) if ((x != blue_pos_flat) and (x != red_pos_flat))]
    )
    flat_coin_pos = np.random.choice(possible_coin_pos)
    return _unflatten_index(flat_coin_pos, grid_size)


@jit(nopython=True)
def _flatten_index(pos, grid_size):
    y_pos, x_pos = pos
    idx = grid_size * y_pos
    idx += x_pos
    return idx


@jit(nopython=True)
def _unflatten_index(pos, grid_size):
    x_idx = pos % grid_size
    y_idx = pos // grid_size
    return np.array([y_idx, x_idx])


@jit(nopython=True)
def generate_observations_wt_numba_optimization(
    batch_size, red_pos, blue_pos, coin_pos, red_coin, grid_size
):
    obs = np.zeros((batch_size, grid_size, grid_size, 4))
    for i in prange(batch_size):
        obs[i, red_pos[i][0], red_pos[i][1], 0] = 1
        obs[i, blue_pos[i][0], blue_pos[i][1], 1] = 1
        if red_coin[i]:
            obs[i, coin_pos[i][0], coin_pos[i][1], 2] = 1
        else:
            obs[i, coin_pos[i][0], coin_pos[i][1], 3] = 1
    return obs
