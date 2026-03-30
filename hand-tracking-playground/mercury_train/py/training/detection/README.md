# Hand Detector!

🚧🚧 Under construction! 🚧🚧

If you need help quickly, contact me:

mailto:moses@collabora.com

meowses#6942 (discord)

# Getting set up:

## Clone HMDHandRects somewhere:

```
git clone https://gitlab.collabora.com/moses/hmdhandrects.git
```

On some distros you may need to do this:
```
git lfs install
git lfs fetch
git lfs pull
```
to get the image files to download.

It should take an hour or so to download everything (~50gb)

Make a hardlink for yourself:
```
ln -sv /path/to/HMDHandRects/ datasets/HMDHandRects
```

## todo: EgoHands and EPIC-KITCHENS

# Running:
```
python3 trainer_detection.py
```