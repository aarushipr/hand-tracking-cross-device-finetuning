# Todos

## Short term

- Create a keypoint-based semi-automatic annotator based on the old bounding-box-based annotator
- Collect+annotate data for hand detection from Reverb/Index.
Hand detector is currently underfit to portrait/square images and thinks hands are bigger than they really are.
- Put the hand detector trainer code in here
- Update SteamVR driver and get it on Steam
- Share optimizer code blocks between data generator and inference code
- Optimizer estimate own variances

## Longer term

- Implement a true random walk/verlet integration for data generation
- Generate artificial data with overlapping hands
  - Should be able to use the optimizer to stop hands from phasing through each other (need verlet integration first!) and cause complicated intertwining behavour by telling joints on specific fingers to be close/far apart
- Hand detector that runs on eg. the past three frames so it can use motion to make better guesses
- Better architectures in keypoint estimators.
