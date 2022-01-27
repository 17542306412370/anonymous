# import random
# from typing import Dict, List, Any, Union
#
# from ray.rllib.policy.sample_batch import SampleBatch
# from ray.rllib.utils.framework import TensorType
# from ray.rllib.utils.typing import ModelInputDict
# from ray.rllib.models.modelv2 import flatten, restore_original_dimensions
#
#
# def after_init_wrap_model_other_play(
#     policy, obs_space, action_space, config, symetries_available
# ):
#     print("after init other play")
#     policy.model = OtherPlayModeleWrapper(policy.model, symetries_available)
#     policy.unwrapped_model = OtherPlayModeleWrapper(
#         policy.unwrapped_model, symetries_available
#     )
#     if hasattr(policy, "target_q_model"):
#         policy.target_q_model = OtherPlayModeleWrapper(
#             policy.target_q_model, symetries_available
#         )
#
#
# class OtherPlayModeleWrapper:
#     def __init__(self, model, symetries_available):
#         self._original_model = model
#         self._symetries_available = symetries_available
#         self.select_new_symmetry()
#
#     def __getattr__(self, item):
#         print("item", item)
#         # print(
#         #     "object.__getattribute__(self, __dict__).keys()",
#         #     object.__getattribute__(self, "__dict__").keys(),
#         # )
#         # if item in dir(object.__getattribute__(self, "__dict__").keys():
#         return self._original_model.__getattribute__(item)
#         # else:
#         #     return object.__getattribute__(self, item)
#
#     def forward(
#         self,
#         input_dict: Dict[str, TensorType],
#         state: List[TensorType],
#         seq_lens: TensorType,
#     ) -> (TensorType, List[TensorType]):
#         print("forward")
#         self.select_new_symmetry()
#
#         print("input_dict", input_dict)
#         print(
#             "input before",
#             input_dict["obs_flat"].shape,
#             input_dict["obs_flat"],
#         )
#         input_dict["obs_flat"] = self._symmetry_in_use["obs_sym"](
#             input_dict["obs_flat"]
#         )
#         print("input after", input_dict["obs_flat"])
#         logits, state = self._original_model.forward(
#             input_dict, state, seq_lens
#         )
#
#         print("logits before", logits.shape, logits)
#         logits = self._symmetry_in_use["act_sym"](logits)
#         print("logits after", logits)
#         return logits, state
#
#     def __call__(
#         self,
#         input_dict: Union[SampleBatch, ModelInputDict],
#         state: List[Any] = None,
#         seq_lens: TensorType = None,
#     ) -> (TensorType, List[TensorType]):
#         """
#         Clone of ray.rllib.models.modelv2
#         """
#
#         # Original observations will be stored in "obs".
#         # Flattened (preprocessed) obs will be stored in "obs_flat".
#
#         # SampleBatch case: Models can now be called directly with a
#         # SampleBatch (which also includes tracking-dict case (deprecated now),
#         # where tensors get automatically converted).
#         if isinstance(input_dict, SampleBatch):
#             restored = input_dict.copy(shallow=True)
#             # Backward compatibility.
#             if seq_lens is None:
#                 seq_lens = input_dict.seq_lens
#             if not state:
#                 state = []
#                 i = 0
#                 while "state_in_{}".format(i) in input_dict:
#                     state.append(input_dict["state_in_{}".format(i)])
#                     i += 1
#             input_dict["is_training"] = input_dict.is_training
#         else:
#             restored = input_dict.copy()
#         restored["obs"] = restore_original_dimensions(
#             input_dict["obs"],
#             self._original_model.obs_space,
#             self._original_model.framework,
#         )
#         try:
#             if len(input_dict["obs"].shape) > 2:
#                 restored["obs_flat"] = flatten(
#                     input_dict["obs"], self._original_model.framework
#                 )
#             else:
#                 restored["obs_flat"] = input_dict["obs"]
#         except AttributeError:
#             restored["obs_flat"] = input_dict["obs"]
#         with self._original_model.context():
#             res = self.forward(restored, state or [], seq_lens)
#         if (not isinstance(res, list) and not isinstance(res, tuple)) or len(
#             res
#         ) != 2:
#             raise ValueError(
#                 "forward() must return a tuple of (output, state) tensors, "
#                 "got {}".format(res)
#             )
#         outputs, state_out = res
#
#         if not isinstance(state_out, list):
#             raise ValueError(
#                 "State output is not a list: {}".format(state_out)
#             )
#
#         self._original_model._last_output = outputs
#         return outputs, state_out if len(state_out) > 0 else (state or [])
#
#     def select_new_symmetry(self):
#         self._symmetry_in_use = random.choice(self._symetries_available)
