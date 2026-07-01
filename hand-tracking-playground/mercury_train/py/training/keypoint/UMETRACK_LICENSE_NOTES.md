# Licensing notes: UmeTrack / HOT3D / hand_tracking_toolkit

Before running `convert_umetrack_to_rando_csv.py` on real downloaded data,
be aware of the license stack involved. There are three separate licenses
in play, not one:

## 1. `hand_tracking_toolkit` (the code)
Apache 2.0. Permissive, no restriction on commercial use. This is the
loader/conversion code itself (`build_hand_dataset`, camera models,
`umetrack_hand_model.forward_kinematics`, etc.) -- safe to vendor/depend on
however you like.

## 2. The UmeTrack dataset (the data)
**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial), per
[facebookresearch/UmeTrack_data's LICENSE](https://github.com/facebookresearch/UmeTrack_data).
Non-commercial only. This means:
- Fine for this thesis (academic, non-commercial use).
- NOT fine to use for training a model that ends up in a commercial
  product, even indirectly (e.g. a checkpoint pretrained partly on
  UmeTrack-derived data, later fine-tuned and shipped commercially, is
  generally considered a derivative work under CC BY-NC).
- Attribution required if the dataset/results are published (cite the
  UmeTrack SIGGRAPH Asia 2022 paper -- see hand_tracking_toolkit's README
  for the BibTeX).

## 3. The HOT3D dataset (the data)
Gated behind Meta's own dataset license at
https://www.projectaria.com/datasets/hot3d/license/ (requires reviewing
and accepting terms on Meta's site -- this document does not attempt to
restate those terms, since they can change; read the current version
before downloading). Historically Meta's research dataset licenses in
this space have also been non-commercial/research-only, similar in spirit
to UmeTrack's CC BY-NC, but confirm directly rather than assuming.

## 4. MANO (optional, only if you use MANO-format annotations)
Requires a separate signup/license at https://mano.is.tue.mpg.de/ (via
smpl-x.is.tue.mpg.de per the toolkit's README) and installing the
third-party `smplx`/`chumpy` packages. This converter does not use MANO
annotations (it uses the UmeTrack hand model directly, which the toolkit's
own README recommends anyway: "For training better models, we recommend
using the UmeTrack annotations"), so MANO's license does not apply to this
specific pipeline unless that's changed later.

## Practical implication for this thesis

Everything above is compatible with finishing and submitting the thesis
on the original timeline. The constraint only bites if/when there's ever
a commercial product built on top of this work:

- Track provenance: keep any checkpoint trained using UmeTrack (or
  HOT3D) data clearly labeled/separated from checkpoints trained only on
  this project's own synthetic (Blender-generated) data plus whichever of
  Panoptic/FreiHand/etc. have compatible licenses.
- Before any commercial use, either: (a) drop UmeTrack/HOT3D-derived data
  and retrain from a checkpoint that never saw it, or (b) get explicit
  permission from Meta for that use case.
- The synthetic Blender pipeline (`mercury_train`'s own data generator)
  has no such restriction -- it's fully your own generated data -- so it
  remains the safe default to lean on for anything beyond the thesis.

None of this blocks using UmeTrack for the thesis itself; it just means
"free data now, throw it away later if this goes commercial" is the
correct mental model, exactly as flagged in the original supervisor
conversation.
