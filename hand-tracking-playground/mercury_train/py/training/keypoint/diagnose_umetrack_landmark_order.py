"""
Definitively resolve UmeTrack's 20-landmark order by walking the hand
skeleton's actual kinematic tree, instead of guessing from images.

convert_umetrack_to_rando_csv.py currently assumes UmeTrack's 20 landmarks
are laid out as 5 fingers x 4 joints in thumb/index/middle/ring/pinky order
(see UMETRACK_TO_CANONICAL_LANDMARK_MAPPING assumption in that file). A
visual check of converted crops (verify_umetrack_visual.py) looked
suspicious -- the assumed "thumb" trace kept landing near the top of the
hand alongside the other fingertips across multiple samples, rather than
clearly separated as its own digit, across multiple different frames. That
pattern is consistent with the assumed order being wrong.

Rather than trust more eyeballing, this walks the actual skeleton data:
- joint_parent / joint_first_child / joint_next_sibling define the real
  bone hierarchy (root -> wrist -> 5 finger chains of 3 frames each).
- landmark_rest_bone_indices says which bone each of the 20 landmarks is
  primarily skinned to (via argmax of landmark_rest_bone_weights).

Walking the tree from the wrist's first child, then following
next_sibling repeatedly, gives the five finger-chain root frames in their
TRUE order as stored in this specific hand model -- no assumption needed.
Landmarks are then grouped by which finger-chain frame they're closest to
(by rest-pose 3D distance, as a tie-breaker within a finger), which
recovers the finger groupings and their real order.

Usage (run on the cluster, same place the .tar files were downloaded):
    python diagnose_umetrack_landmark_order.py \
        --root /storage/user/praa/umetrack_sample/train \
        --sequences subject_000_separate_hand_000000
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--sequences", required=True, nargs="+")
    args = parser.parse_args()

    from hand_tracking_toolkit.dataset import build_hand_dataset

    dataset = build_hand_dataset(
        args.root, args.sequences,
        load_monochrome=True, load_rgb=False,
        output_crops=True, crop_size=128,
    )

    hand_model = None
    for sample_crops in dataset:
        for crop in sample_crops:
            if crop.hand_shape is not None and crop.hand_shape.umetrack is not None:
                hand_model = crop.hand_shape.umetrack
                break
        if hand_model is not None:
            break

    if hand_model is None:
        print("Couldn't find a crop with hand_shape data -- try more/different --sequences.")
        return

    joint_parent = hand_model.joint_parent.cpu().numpy()
    joint_first_child = hand_model.joint_first_child.cpu().numpy()
    joint_next_sibling = hand_model.joint_next_sibling.cpu().numpy()
    landmark_rest_positions = hand_model.landmark_rest_positions.cpu().numpy()  # (20, 3)
    bone_weights = hand_model.landmark_rest_bone_weights.cpu().numpy()
    bone_indices = hand_model.landmark_rest_bone_indices.cpu().numpy()
    joint_rest_positions = hand_model.joint_rest_positions.cpu().numpy()

    print(f"num joint frames: {len(joint_parent)}, num landmarks: {len(landmark_rest_positions)}")
    print(f"joint_parent: {joint_parent.tolist()}")
    print(f"joint_first_child: {joint_first_child.tolist()}")
    print(f"joint_next_sibling: {joint_next_sibling.tolist()}")
    print()

    # NUM_JOINT_FRAMES = 1 (root) + 1 (wrist) + 3*5 (finger frames) = 17
    # Frame 0 = root, frame 1 = wrist (per umetrack_hand_model.py's
    # NUM_JOINT_FRAMES comment). Walk wrist's children to get the five
    # finger-chain roots IN THE ORDER THE MODEL ACTUALLY STORES THEM.
    WRIST_FRAME = 1
    finger_chain_roots = []
    child = joint_first_child[WRIST_FRAME]
    while child != -1:
        finger_chain_roots.append(int(child))
        child = joint_next_sibling[child]

    print(f"Finger chain root frames, in the model's real sibling order: {finger_chain_roots}")
    print(f"(expected 5 entries for 5 fingers, got {len(finger_chain_roots)})")
    print()

    # For each finger chain root, walk down first_child to get all 3 frames
    # in that chain (mcp, pip, dip -- the tip landmark has no rotational
    # frame of its own, it's a fixed offset skinned to the dip frame).
    finger_frame_chains = []
    for root_frame in finger_chain_roots:
        chain = [root_frame]
        f = joint_first_child[root_frame]
        while f != -1:
            chain.append(int(f))
            f = joint_first_child[f]
        finger_frame_chains.append(chain)
        print(f"  chain starting at frame {root_frame}: {chain}")
    print()

    # Assign each of the 20 landmarks to whichever bone it's MOST skinned
    # to (argmax weight), then to whichever finger chain contains that
    # bone. This directly answers "which 4 landmarks belong to finger N,
    # and in this model's real order, which finger is N?" with no
    # assumption about thumb/index/middle/ring/pinky ordering at all.
    primary_bone = np.array([
        bone_indices[i, np.argmax(bone_weights[i])] for i in range(len(landmark_rest_positions))
    ])

    print("Landmark -> finger-chain-index assignment (0-indexed by discovery order above):")
    for finger_i, chain in enumerate(finger_frame_chains):
        landmark_ids = [i for i in range(20) if primary_bone[i] in chain]
        # Order landmarks within the finger by distance from the chain
        # root's rest position (proxy for mcp -> pip -> dip -> tip order).
        root_pos = joint_rest_positions[chain[0]]
        landmark_ids.sort(key=lambda i: np.linalg.norm(landmark_rest_positions[i] - root_pos))
        print(f"  finger chain {finger_i} (frames {chain}): landmarks {landmark_ids}")

    print()
    print("Compare this grouping/order against the assumption in")
    print("convert_umetrack_to_rando_csv.py (currently: landmarks 0-3=thumb, 4-7=index,")
    print("8-11=middle, 12-15=ring, 16-19=pinky, each in mcp->pip->dip->tip order).")
    print("If the groupings above don't match that in both membership AND order,")
    print("umetrack_landmarks_to_project_keypoints() needs its permutation updated")
    print("to match what's printed here.")


if __name__ == "__main__":
    main()
