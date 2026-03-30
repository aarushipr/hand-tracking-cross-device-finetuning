from dataclasses import dataclass


@dataclass
class aug_config(object):
  # List of tuples
  radius_function_prediction: list
  radius_function_no_prediction: list
  move_amount_predicted: float
  move_amount_normal: float
  output_size: int


radius_function_predicted = [
    (0.9, 0.1), (1.2, 1.0), (1.4, 0.9), (1.6, 0.8), (1.8, 0.5), (2.0, 0.2), (2.4, 0.0)]
radius_function_not_predicted = [
    (0.9, 0.1), (1.2, 1.0), (1.4, 0.9), (1.6, 0.8), (1.8, 0.5), (2.0, 0.2), (2.4, 0.0)]

the_aug_config = aug_config(
    radius_function_predicted, radius_function_not_predicted, 0.01, 0.5, 128)
