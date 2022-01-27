import logging
import random
from abc import ABC

import numpy as np

from submission.envs.utils.interfaces import InfoAccumulationInterface

logger = logging.getLogger(__name__)


class TwoPlayersTwoActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in a two player discrete game.
    Logs the frequency of each state.
    """

    def _init_info(self):
        self.cc_count = []
        self.dd_count = []
        self.cd_count = []
        self.dc_count = []

    def _reset_info(self):
        self.cc_count.clear()
        self.dd_count.clear()
        self.cd_count.clear()
        self.dc_count.clear()

    def _get_episode_info(self):
        return {
            "CC_freq": sum(self.cc_count) / len(self.cc_count),
            "DD_freq": sum(self.dd_count) / len(self.dd_count),
            "CD_freq": sum(self.cd_count) / len(self.cd_count),
            "DC_freq": sum(self.dc_count) / len(self.dc_count),
        }

    def _accumulate_info(self, ac0, ac1):
        self.cc_count.append(ac0 == 0 and ac1 == 0)
        self.cd_count.append(ac0 == 0 and ac1 == 1)
        self.dc_count.append(ac0 == 1 and ac1 == 0)
        self.dd_count.append(ac0 == 1 and ac1 == 1)


class NPlayersNDiscreteActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in N player games with discrete
    actions.
    Logs the frequency of action profiles used
    (action profile: the set of actions used during one step by all players).
    """

    def _init_info(self):
        self.info_counters = {"n_steps_accumulated": 0}

    def _reset_info(self):
        self.info_counters = {"n_steps_accumulated": 0}

    def _get_episode_info(self):
        info = {}
        if self.info_counters["n_steps_accumulated"] > 0:
            for k, v in self.info_counters.items():
                if k != "n_steps_accumulated":
                    info[k] = v / self.info_counters["n_steps_accumulated"]

        return info

    def _accumulate_info(self, *actions):
        id = "_".join([str(a) for a in actions])
        if id not in self.info_counters:
            self.info_counters[id] = 0
        self.info_counters[id] += 1
        self.info_counters["n_steps_accumulated"] += 1


class NPlayersNContinuousActionsInfoMixin(InfoAccumulationInterface, ABC):
    """
    Mixin class to add logging capability in N player games with continuous
    actions.
    Logs the mean and std of action profiles used
    (action profile: the set of actions used during one step by all players).
    """

    def __init__(self, *arg, **kwargs):
        logger.warning(
            "MIXING NPlayersNContinuousActionsInfoMixin NOT DEBBUGED, NOT TESTED"
        )
        super().__init__(*arg, **kwargs)

    def _init_info(self):
        self.data_accumulated = {}

    def _reset_info(self):
        self.data_accumulated = {}

    def _get_episode_info(self):
        info = {}
        for k, v in self.data_accumulated.items():
            array = np.array(v)
            info[f"{k}_mean"] = array.mean()
            info[f"{k}_std"] = array.std()

        return info

    def _accumulate_info(self, **kwargs_actions):
        for k, v in kwargs_actions.items():
            if k not in self.data_accumulated.keys():
                self.data_accumulated[k] = []
            self.data_accumulated[k].append(v)


class OtherPlayMixin:
    def _select_new_symmetries(self):
        if self._use_other_play:
            self._symmetries_in_use = [
                random.choice(self.SYMMETRIES) for _ in range(self.NUM_AGENTS)
            ]

    def _apply_other_play(self, rllib_obs_or_actions, obs=False, acts=False):
        if self._use_other_play:
            assert (obs + acts) == 1
            # print(
            #     "rllib_obs_or_actions before", "obs", obs, rllib_obs_or_actions
            # )
            rllib_obs_or_actions = {
                agent_id: symmetry_fn["obs_sym"](v)
                if obs
                else symmetry_fn["act_sym"](v)
                for (agent_id, v), symmetry_fn in zip(
                    rllib_obs_or_actions.items(), self._symmetries_in_use
                )
            }
            # print(
            #     "rllib_obs_or_actions after", "obs", obs, rllib_obs_or_actions
            # )
        return rllib_obs_or_actions
