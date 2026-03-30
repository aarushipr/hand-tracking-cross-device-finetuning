import numpy as np
import cv2
import matplotlib
import random
import math


def mat_float32touint8(mat):
  return (mat*255).astype(np.uint8)


def mat_uint8tofloat32(mat):
  return mat.astype(np.float32)*1.0/255.0


# based very loosely on https://stackoverflow.com/a/49848093
def make_radial_gradient(sz, pos_x, pos_y, radius, softness=1.0):
  # Generate (x,y) coordinate arrays
  y, x = np.mgrid[0:sz, 0:sz]
  # Calculate the weight for each pixel
  distance_squared = ((x-pos_x)**2 + (y-pos_y)**2)

  return np.clip((radius**2 - distance_squared) / (softness*radius) + 0.5, 0, 1).astype(np.float32)


def interize(b):
  return (int(b[0]), int(b[1]))


def transformVecBy2x3(input, M):
  thing = input.copy()
  thing[0] = (input[0] * M[0, 0]) + (input[1] * M[0, 1]) + M[0, 2]
  thing[1] = (input[0] * M[1, 0]) + (input[1] * M[1, 1]) + M[1, 2]

  return thing


def in_line(x, p1, p2):
  return ((p2[1]-p1[1])/(p2[0]-p1[0]))*(x-p1[0])+p1[1]


def piecewise(x, pts_list):
  for i in range(len(pts_list)):
    if (x > pts_list[i][0]) and (x < pts_list[i+1][0]):
      return in_line(x, pts_list[i], pts_list[i+1])
  raise RuntimeError("Input not in list of points!")


def arbitrary_density(val_and_density_list):
  curr_pt = 0
  old_curr_pt = 0

  sum = 0

  for i in range(len(val_and_density_list)-1):
    sum += val_and_density_list[i][1]
  x = random.uniform(0, sum)

  for i in range(len(val_and_density_list)-1):
    old_curr_pt = curr_pt
    curr_pt += val_and_density_list[i][1]
    if (x < curr_pt):
      the_val = in_line(
          x, (old_curr_pt, val_and_density_list[i][0]), (curr_pt, val_and_density_list[i+1][0]))
      return the_val

  raise RuntimeError("Input not in list of points!")


def draw_hand_rainbow_pts(img, pts):
  for j in range(21):
    rgb = matplotlib.colors.hsv_to_rgb([j/21.0, 1.0, 1.0])
    for ele in rgb:
      ele *= 255.0
    cv2.circle(img, (int(pts[j][0]), int(pts[j][1])),
               5, (rgb[2], rgb[1], rgb[0]), cv2.FILLED)


def calc_num_times_to_make_image(requested_num_images, num_unique_hands):
  base_num = math.floor(requested_num_images / num_unique_hands)
  extra_probability = (requested_num_images %
                       num_unique_hands) / num_unique_hands
  am = base_num
  if (random.random() < extra_probability):
    am += 1
  return am


def npImgWidth(img):
  return img.shape[1]


def npImgHeight(img):
  return img.shape[0]


def normalizeGrayscaleImage(img, report=None,  target_mean=0.5, target_std=0.25):
  if img.dtype == np.uint8:
    img = mat_uint8tofloat32(img)
  std = np.std(img)
  if std < 0.0001:
    if report is not None:
      print(f"Very low contrast: {report}")
    img = np.random.random(img.shape)
    std = np.std(img)
  img *= target_std / std
  img += target_mean - np.mean(img)

  np.clip(img, 0, 1)

  return img


def draw_rectangle_in_image_px_coord(image, top, bottom, left, right, color):
  # I wish I had any idea why the normal cv2.rectangle() crashes my machine.
  # like it legit crashes my GPU and I have to restart my computer.
  cv2.line(image,
           (int(left), int(top)),
           (int(right), int(top)),
           color, thickness=1)
  cv2.line(image,
           (int(right), int(top)),
           (int(right), int(bottom)),
           color, thickness=1)
  cv2.line(image,
           (int(right), int(bottom)),
           (int(left), int(bottom)),
           color, thickness=1)
  cv2.line(image,
           (int(left), int(bottom)),
           (int(left), int(top)),
           color, thickness=1)


def draw_square_in_image_px_coord(image, center, square_side, color):
  draw_rectangle_in_image_px_coord(
      image, center[1] - (square_side / 2),
      center[1] + (square_side / 2),
      center[0] - (square_side / 2),
      center[0] + (square_side / 2),
      color)


def draw_rectangle_in_hmap(hmap, top, bottom, left, right, val):
  width = hmap.shape[1]
  height = hmap.shape[0]

  top = int(top * height)
  bottom = int(bottom * height)

  left = int(left * width)
  right = int(right * width)

  hmap[top:bottom, left:right] = val


def darknet_to_square_in_image_px_coord(image_w, image_h, cx, cy, w, h):
  center = (cx*image_w, cy*image_h)
  square_size = max(w*image_w, h*image_h)
  return center[0], center[1], square_size

# 2D Miniball


def minicircle(pts_):
  pts = pts_[:, :2]
  min_x = pts[0][0]
  min_y = pts[0][1]

  max_x = pts[0][0]
  max_y = pts[0][1]

  # Find axis-aligned bounding box.
  # This is our "close enough" to the center of the smallest enclosing circle
  for pt in pts:
    min_x = min(min_x, pt[0])
    max_x = max(max_x, pt[0])

    min_y = min(min_y, pt[1])
    max_y = max(max_y, pt[1])

  cx = (min_x + max_x) / 2
  cy = (min_y + max_y) / 2

  c = np.array([cx, cy])

  # Find radius
  r = 0
  for pt in pts:
    r = max(r, np.linalg.norm(c-pt))

  return cx, cy, r


def rgb_imshow(name, img):
  new_img = cv2.cvtcolor(img, cv2.COLOR_BGR2RGB)
  cv2.imshow(name, new_img)