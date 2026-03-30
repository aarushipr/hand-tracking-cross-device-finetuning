# Monado's hand tracking!

🚧🚧 This wiki is under construction! 🚧🚧

If you need help quickly, contact me:

mailto:moses@collabora.com

meowses#6942 (discord)




# Hand detector

## Datasets

[HMDHandRects](https://gitlab.collabora.com/moses/hmdhandrects)

[EgoHands](http://domedb.perception.cs.cmu.edu/handdb.html)

[EPIC-KITCHENS](https://epic-kitchens.github.io/2022)

## [Annotation code](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-playground/hmdhandrects-annotator)

## [Training code](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-playground/hand-detector)


# Keypoint estimator

## [Artificial dataset generator](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-playground/image-distorter)

## Other datasets

[FreiHand](https://github.com/lmb-freiburg/freihand)

[CMU Panoptic Dataset](http://domedb.perception.cs.cmu.edu/handdb.html)

## [Training code](https://gitlab.freedesktop.org/monado/utilities/hand-tracking-playground/keypoint-estimator)


# Inference implementations

## [Default implementation, inside of Monado](https://gitlab.freedesktop.org/monado/monado/-/tree/main/src/xrt/tracking/hand/mercury)

## [SteamVR driver (Valve Index only)](https://github.com/slitcch/mercury_steamvr_driver)