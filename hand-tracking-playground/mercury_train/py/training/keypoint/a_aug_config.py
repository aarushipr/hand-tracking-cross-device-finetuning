from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class radius_move:
    radius_function: List[Tuple]
    move_amount_function: List[Tuple]


@dataclass
class aug_config(object):
    # List of tuples
    #   radius_function_prediction: list
    #   radius_function_no_prediction: list
    #   move_amount_function_prediction: list
    #   move_amount_function_no_prediction: list
    validation_dataset: bool
    output_size: int
    not_hand_likelihood: float


the_aug_config = aug_config(
    validation_dataset=False,
    output_size=128,
    not_hand_likelihood=0.015)  # 1.5% not-hands


# 0% not hands. (Except for the one validation dataset where we'll make
# them 100% not-hands.)
aug_config_validatoor = aug_config(
    validation_dataset=True,
    output_size=128,
    not_hand_likelihood=0)


# 0% not hands. (Except for the one validation dataset where we'll make
# them 100% not-hands.)
aug_config_validatoor_not_hand = aug_config(
    validation_dataset=True,
    output_size=128,
    not_hand_likelihood=1)
