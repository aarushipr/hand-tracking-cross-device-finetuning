// Copyright 2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  StereoKit visualization stuff!
 * @author Moses Turner <mosesturner@protonmail.com>
 */
#pragma once

#ifndef VIZ_USE_XRT
#define VIZ_USE_XRT
#endif

#ifdef VIZ_USE_XRT
#include "math/m_api.h"
#include "math/m_vec3.h"
#include "xrt/xrt_defines.h"
#endif
// actually - meh, just use Monado.

#include "stereokit.h"
#include "stereokit_ui.h"
#include <string>

static enum xrt_hand_joint joints_5x5_to_26[5][5] = {
    {
        XRT_HAND_JOINT_WRIST,
        XRT_HAND_JOINT_THUMB_METACARPAL,
        XRT_HAND_JOINT_THUMB_PROXIMAL,
        XRT_HAND_JOINT_THUMB_DISTAL,
        XRT_HAND_JOINT_THUMB_TIP,
    },
    {
        XRT_HAND_JOINT_INDEX_METACARPAL,
        XRT_HAND_JOINT_INDEX_PROXIMAL,
        XRT_HAND_JOINT_INDEX_INTERMEDIATE,
        XRT_HAND_JOINT_INDEX_DISTAL,
        XRT_HAND_JOINT_INDEX_TIP,
    },
    {
        XRT_HAND_JOINT_MIDDLE_METACARPAL,
        XRT_HAND_JOINT_MIDDLE_PROXIMAL,
        XRT_HAND_JOINT_MIDDLE_INTERMEDIATE,
        XRT_HAND_JOINT_MIDDLE_DISTAL,
        XRT_HAND_JOINT_MIDDLE_TIP,
    },
    {
        XRT_HAND_JOINT_RING_METACARPAL,
        XRT_HAND_JOINT_RING_PROXIMAL,
        XRT_HAND_JOINT_RING_INTERMEDIATE,
        XRT_HAND_JOINT_RING_DISTAL,
        XRT_HAND_JOINT_RING_TIP,
    },
    {
        XRT_HAND_JOINT_LITTLE_METACARPAL,
        XRT_HAND_JOINT_LITTLE_PROXIMAL,
        XRT_HAND_JOINT_LITTLE_INTERMEDIATE,
        XRT_HAND_JOINT_LITTLE_DISTAL,
        XRT_HAND_JOINT_LITTLE_TIP,
    }};

static const char *kine_keys[5][5] = {

    {
        "WRIST",
        "THMB_MCP",
        "THMB_PXM",
        "THMB_DST",
        "THMB_TIP",
    },

    {
        "INDX_MCP",
        "INDX_PXM",
        "INDX_INT",
        "INDX_DST",
        "INDX_TIP",
    },

    {
        "MIDL_MCP",
        "MIDL_PXM",
        "MIDL_INT",
        "MIDL_DST",
        "MIDL_TIP",
    },

    {
        "RING_MCP",
        "RING_PXM",
        "RING_INT",
        "RING_DST",
        "RING_TIP",
    },

    {
        "LITL_MCP",
        "LITL_PXM",
        "LITL_INT",
        "LITL_DST",
        "LITL_TIP",
    },
};

using namespace sk;

inline sk::vec3 hand_to_head(handed_ handed) {
  return input_head()->position - input_hand(handed)->palm.position;
}

#ifdef VIZ_USE_XRT
inline bool palm_facing_head(handed_ handed) {
  const sk::hand_t *hand = input_hand(handed);
  if (!(hand->tracked_state & button_state_active)) {
    return false;
  }

  xrt_vec3 palm_direction;
  math_quat_rotate_vec3((xrt_quat *)&hand->palm.orientation,
                        (xrt_vec3 *)&vec3_forward, (xrt_vec3 *)&palm_direction);
  palm_direction = m_vec3_normalize(palm_direction);
  sk::vec3 hand_to_head_direction = sk::vec3_normalize(hand_to_head(handed));

  return m_vec3_dot(palm_direction, *(xrt_vec3 *)&hand_to_head_direction) >
         0.5f;
}
#endif

static void _draw_axis(const matrix &mat, float size, float thickness) {
  sk::vec3 pos = matrix_extract_translation(mat);
  line_add(pos, matrix_transform_dir(mat, {1, 0, 0}) * size + pos,
           {255, 0, 0, 255}, {255, 0, 0, 255}, thickness);
  line_add(pos, matrix_transform_dir(mat, {0, 1, 0}) * size + pos,
           {0, 255, 0, 255}, {0, 255, 0, 255}, thickness);
  line_add(pos, matrix_transform_dir(mat, {0, 0, 1}) * size + pos,
           {0, 0, 255, 255}, {0, 0, 255, 255}, thickness);
}

// Thickness is as a factor of size.
static void draw_axis(const pose_t &pose, float size = 0.1f,
                      float thickness = 0.1f) {
  // matrix mat = ;
  _draw_axis(pose_matrix(pose), size, thickness * size);
}

static void draw_axis(const sk::vec3 &position, float size = 0.1f,
                      float thickness = 0.01f) {
  _draw_axis(matrix_trs(position), size, thickness);
}

static bool draw_hand_axes() {
  for (int i = 0; i < handed_max; i++) {
    const hand_t *hand = input_hand((handed_)i);
    if (hand->tracked_state == button_state_inactive) {
      continue;
    }

    for (int finger = 0; finger < 5; finger++) {
      for (int joint = 0; joint < 5; joint++) {
        hand_joint_t joint_ = hand->fingers[finger][joint];
        pose_t joint_pose = {joint_.position, joint_.orientation};

        draw_axis(joint_pose, 0.015f, 0.002);
      }
    }

    pose_t palm_pose = hand->palm;
    draw_axis(palm_pose, 0.015f, 0.002);

    pose_t wrist_pose = hand->wrist;
    // log_diagf("I think wrist is at %f %f %f", wrist_pose.position.x,
    // wrist_pose.position.y, wrist_pose.position.z);
    draw_axis(wrist_pose, 15.0f, 0.002);
    vec3 wrist = wrist_pose.position;
    vec3 middle_distal = hand->fingers[2][1].position;
    vec3 sub = wrist - middle_distal;
  }
  return true;
}

static void draw_hand_lines() {
  for (int side = 0; side < handed_max; side++) {
    const hand_t *hand = input_hand((handed_)side);
    if (hand->tracked_state == button_state_inactive)
      continue;

    for (int finger = 0; finger < 5; finger++) {
      for (int joint = 0; joint < 4; joint++) {
        hand_joint_t joint0 = hand->fingers[finger][joint];
        hand_joint_t joint1 = hand->fingers[finger][joint + 1];
        float hue0 = ((finger * 5) + joint) / 24.0f;
        float hue1 = ((finger * 5) + joint + 1) / 24.0f;
        if (finger == 0 && joint == 0) {
          line_add(hand->wrist.position, joint0.position,
                   color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
                   color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.002);
        } else {
          line_add(joint0.position, joint1.position,
                   color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
                   color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.002);
        }
      }
    }
  }
}

static void text_from_vec3(sk::vec3 at, const char *hi) {
  vec3 left_position = hierarchy_to_world_point(at);

  hierarchy_set_enabled(false);

  vec3 head_position = input_head()->position;
  pose_t pose;
  pose.position = (left_position * .9) + (head_position * .1);
  pose.orientation = quat_lookat(left_position, head_position);
  text_add_at(hi, pose_matrix(pose));

  hierarchy_set_enabled(true);
}

static void text_from_vec3(sk::vec3 at, float size, const char *hi) {
  vec3 left_position = hierarchy_to_world_point(at);

  hierarchy_set_enabled(false);

  vec3 head_position = input_head()->position;
  pose_t pose;
  pose.position = (left_position * .9) + (head_position * .1);
  pose.orientation =
      input_head()->orientation; // quat_lookat(left_position, head_position);
  text_add_at(hi, pose_matrix(pose, {-size, size, size}));

  hierarchy_set_enabled(true);
}

static void hand_window(sk::handed_ hand, const char *hi) {
  const hand_t *hande = input_hand(hand);
  if (hande->tracked_state == button_state_inactive)
    return;
  vec3 left_position = hande->palm.position;

  vec3 head_position = input_head()->position;
  pose_t pose;
  pose.position = (left_position * .9) + (head_position * .1);
  pose.orientation = quat_lookat(left_position, head_position);
  // quat_mul(head_potition, quat_from_angles(0, 180, 0)); // head_potition;
  if (input_hand(hand)->tracked_state == button_state_active) {
    hierarchy_push(pose_matrix(pose));
    text_add_at(hi, matrix_identity);
    hierarchy_pop();
    // ui_window_begin(hi, pose, vec2{0, 0});
    // ui_window_end();
  }
}

static void ruler_window() {
  static pose_t window_pose = pose_t{{0, 0, 0.5f}, quat_identity};
  ui_handle_begin("Ruler", window_pose,
                  bounds_t{vec3_zero, vec3{30 * cm2m, 4 * cm2m, 1 * cm2m}},
                  true, ui_move_exact);
  color32 color = color_to_32(color_hsv(0.6f, 0.5f, 1, 1));
  text_add_at("Centimeters",
              matrix_trs(vec3{14.5f * cm2m, -1.5f * cm2m, -0.6f * cm2m},
                         quat_identity, vec3{0.3f, 0.3f, 0.3f}),
              0, text_align_bottom_left);
  for (int d = 0; d <= 60; d++) {
    float x = d / 2.0f;
    float size = (d % 2 == 0) ? 1.0f : 0.15f;
    line_add(vec3{(15 - x) * cm2m, 2 * cm2m, -0.6f * cm2m},
             vec3{(15 - x) * cm2m, (2 - size) * cm2m, -0.6f * cm2m}, color,
             color, 0.5f * mm2m);

    if (d % 2 == 0 && d / 2 != 30) {
      text_add_at(std::to_string(d / 2).c_str(),
                  matrix_trs(vec3{(15 - x - 0.1f) * cm2m, (2 - size) * cm2m,
                                  -0.6f * cm2m},
                             quat_identity, vec3{0.2f, 0.2f, 0.2f}),
                  0, text_align_bottom_left);
    }
  }
  ui_handle_end();
}

static void draw_camera_axis_at_pose(const sk::vec3 &vec, sk::quat ori,
                                     const char *name) {
  // We could take the matrix and plop it into a sk::matrix,
  // But that's a pretty annoying idea.
  // Basically I REALLY don't trust SK's matrix memory layout, and I kinda don't
  // trust mine. So copying is confusing, I don't want to think about it.
  // Instead, I want to go to a representation that's well-known - just the
  // position and orientation in the OpenXR coordinate space. I trust StereoKit
  // to get this right, I don't totally know if my code gets it right, but by
  // the time I'm done it better.

  sk::pose_t pose;
  pose.position = vec;
  pose.orientation = ori;

  draw_axis(pose, 0.05f);
  text_from_vec3(pose.position, 0.3, name);
}

static void draw_hand_axis_at_pose(const sk::pose_t &pose, const char *name) {
  // We could take the matrix and plop it into a sk::matrix,
  // But that's a pretty annoying idea.
  // Basically I REALLY don't trust SK's matrix memory layout, and I kinda don't
  // trust mine. So copying is confusing, I don't want to think about it.
  // Instead, I want to go to a representation that's well-known - just the
  // position and orientation in the OpenXR coordinate space. I trust StereoKit
  // to get this right, I don't totally know if my code gets it right, but by
  // the time I'm done it better.

  draw_axis(pose, 0.01f);
  text_from_vec3(pose.position, 0.1, name);
}

static sk::pose_t sk_from_xrt(xrt_pose pose) {
  sk::pose_t pose_sk;
  pose_sk.position = *(sk::vec3*)&pose.position;
  pose_sk.orientation = *(sk::quat*)&pose.orientation;
  return pose_sk;
}

static void disp_xrt_hand(xrt_hand_joint_set &hand) {
  // Draw axes
  for (int finger_idx = 0; finger_idx < 5; finger_idx++) {

    // Draw axes
    for (int bone_idx = 0; bone_idx < 5; bone_idx++) {
      int idx = joints_5x5_to_26[finger_idx][bone_idx];

      draw_hand_axis_at_pose(
          sk_from_xrt(hand.values.hand_joint_set_default[idx].relation.pose),
          kine_keys[finger_idx][bone_idx]);
    }

    // Draw lines. Note we're going to only the 4th bone
    for (int bone_idx = 0; bone_idx < 4; bone_idx++) {
      int idx = joints_5x5_to_26[finger_idx][bone_idx];
      int idx1 = joints_5x5_to_26[finger_idx][bone_idx + 1];

      sk::vec3 joint0 =
          sk_from_xrt(hand.values.hand_joint_set_default[idx].relation.pose)
              .position;
      sk::vec3 joint1 =
          sk_from_xrt(hand.values.hand_joint_set_default[idx1].relation.pose)
              .position;

      float hue0 = ((finger_idx * 4) + bone_idx) / 24.0f;
      float hue1 = ((finger_idx * 4) + bone_idx + 1) / 24.0f;
      line_add(joint0, joint1, color_to_32(color_hsv(hue0, 1.0f, 1.0f, 1.0f)),
               color_to_32(color_hsv(hue1, 1.0f, 1.0f, 1.0f)), 0.0005);
    }
  }
}
