import torch.multiprocessing
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import a_geometry as geo

# The below 3 lines are magic, and required to make cv2.imshow() not crash on Arch. Feel free to comment them but please don't remove them
import cv2
cv2.namedWindow('a', 0)
# from matplotlib import pyplot as plt

from torch.utils.data import DataLoader


import sys
import select

import wandb
from CombinedDataset import AllOfTheDatasetsCombined
import KeyNet


torch.autograd.set_detect_anomaly(True)


# https://gitanswer.com/pytorch-too-many-open-files-error-cplusplus-356516297
torch.multiprocessing.set_sharing_strategy('file_system')


class MosesHeatmapLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.float()
    self.criterion = nn.MSELoss()

  def forward(self, pred_xy, gt_xy, attention_xy, pred_depth, gt_depth, attention_depth):

    # Wrong! We should be looking at valids too. Look at whatever you did with epic-kitchens.

    loss = 0
    loss += self.criterion(pred_xy*attention_xy,
                           gt_xy*attention_xy).sum()

    loss += self.criterion(pred_depth*attention_depth,
                           gt_depth*attention_depth).sum()

    return loss*90


class MosesForearmLoss(nn.Module):
  def __init__(self):
    super().__init__()
    self.float()
    self.criterion = nn.MSELoss(reduction="mean")

  def forward(self, model_pred_3d_plus_conf, gt_3d, use_forearm):
    # This is pretty wrong btw, I think
    loss = 0
    loss += self.criterion(model_pred_3d_plus_conf[:, 0] * use_forearm,
                           gt_3d[:, 0]*use_forearm)
    loss += self.criterion(model_pred_3d_plus_conf[:, 1] * use_forearm,
                           gt_3d[:, 1]*use_forearm)
    loss += self.criterion(model_pred_3d_plus_conf[:, 2] * use_forearm,
                           gt_3d[:, 2]*use_forearm)
    return loss*0.1


def find_center_of_distribution(data):
  idx = np.unravel_index(np.argmax(data), data.shape)
  coarse_x = idx[1]
  coarse_y = idx[0]

  w = data.shape[1]
  h = data.shape[0]

  max_kern_width = 3

  # can coarse_x and coarse_y be negative?
  # if not, can we remove the abs?
  kern_width_x = max(0, min(coarse_x, min(
      max_kern_width, abs(coarse_x - w)-1)))
  kern_width_y = max(0, min(coarse_y, min(
      max_kern_width, abs(coarse_y - h)-1)))

  min_x = coarse_x - kern_width_x
  max_x = coarse_x + kern_width_x

  min_y = coarse_y - kern_width_y
  max_y = coarse_y + kern_width_y

  sum_of_values = 0
  sum_of_values_times_locations_x = 0
  sum_of_values_times_locations_y = 0

  for y in range(min_y, max_y+1):
    for x in range(min_x, max_x+1):
      val = data[y][x]
      sum_of_values += val
      sum_of_values_times_locations_y += val * (y + 0.5)
      sum_of_values_times_locations_x += val * (x + 0.5)

  if (sum_of_values == 0):
    print("Ugh, what?")
    return coarse_x, coarse_y

  out_refined_x = sum_of_values_times_locations_x / sum_of_values
  out_refined_y = sum_of_values_times_locations_y / sum_of_values

  return out_refined_x, out_refined_y


def display_output(input_img, gt_hmaps, attention_xy, pred_hmaps, gt_forearm, pred_forearm, use_forearm, gui):

  input_img = (np.clip(input_img, 0, 1) * 255).astype(np.uint8)

  input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

  num = 21
  if use_forearm:
    num = 22

  jointlocs_gt_hmaps = np.zeros((num, 2))
  jointlocs_pred_hmaps = np.zeros((22, 2))

  for i in range(21):
    x, y = find_center_of_distribution(gt_hmaps[i])
    x *= (128/22)
    y *= (128/22)

    jointlocs_gt_hmaps[i][0] = x
    jointlocs_gt_hmaps[i][1] = y

    x, y = find_center_of_distribution(pred_hmaps[i])
    x *= (128/22)
    y *= (128/22)
    jointlocs_pred_hmaps[i][0] = x
    jointlocs_pred_hmaps[i][1] = y

  if use_forearm:
    jointlocs_gt_hmaps[21][0] = gt_forearm[0]
    jointlocs_gt_hmaps[21][1] = gt_forearm[1]

  jointlocs_pred_hmaps[21][0] = pred_forearm[0]
  jointlocs_pred_hmaps[21][1] = pred_forearm[1]

  def draw_lines(thing, forearm, use_forearm_, color, width=1):
    for finger in range(5):
      for joint in range(4):
        idx = 1 + (finger*4) + joint

        previdx = idx-1
        if joint == 0:
          previdx = 0

        prev_gt = thing[previdx]
        curr_gt = thing[idx]

        cv2.line(input_img,
                 (int(prev_gt[0]), int(prev_gt[1])),
                 (int(curr_gt[0]), int(curr_gt[1])),
                 color, width)
    if (use_forearm_):
      prev_gt = thing[0]
      new_forearm = forearm.copy()
      new_forearm[0] *= 30
      new_forearm[1] *= 30
      curr_gt = prev_gt + new_forearm[:2]
      cv2.line(input_img,
               (int(prev_gt[0]), int(prev_gt[1])),
               (int(curr_gt[0]), int(curr_gt[1])),
               color, width)

  def draw_circles(joint_coord, color, valid=np.ones((25, 3))):
    for coord, valid_this in zip(joint_coord, valid):
      if not valid_this.any():
        continue

      cv2.circle(input_img, (int(coord[0]), int(coord[1])), 2, color)

  draw_lines(jointlocs_gt_hmaps, gt_forearm, use_forearm, (0, 255, 0))

  draw_circles(jointlocs_gt_hmaps, (0, 255, 0))
  draw_lines(jointlocs_pred_hmaps, pred_forearm, True, (0, 0, 255))
  draw_circles(jointlocs_pred_hmaps, (0, 0, 255))

  images = [input_img]
  a = np.zeros((22*5, 22*5), np.float32)
  b = np.zeros((22*5, 22*5), np.float32)
  att = np.zeros((22*5, 22*5), np.float32)

  for i in range(21):
    col = (i % 5)*22
    row = (int(i/5))*22
    a[row:row+22, col:col+22] = gt_hmaps[i]
    b[row:row+22, col:col+22] = pred_hmaps[i]
    att[row:row+22, col:col+22] = attention_xy[i]

  images.append(a)
  images.append(b)

  wandb.log({"imgs": [wandb.Image(img) for img in images]})
  if gui:
    geo.rgb_imshow("preedicted joints", input_img)
    cv2.imshow("a a", a)
    cv2.imshow("b", b)
    cv2.imshow("attention", att)

    key = cv2.waitKey(1)
    # Exiting cleanly is important; sometimes you can accidentally ctrl+c when saving the state dict
    if (key == ord('q')):
      exit(0)


def save_checkpoint(states, output_dir, filename='checkpoint.pth'):
  os.makedirs(output_dir, exist_ok=True)
  torch.save(states, os.path.join(output_dir, filename))


def train_loop(device, dataloader, model, optimizer, loss_fn, forearm_loss_fn, gui):
  total_loss = 0
  loss_divisor = 0
  l = len(dataloader)

  for batch, doct in enumerate(dataloader):
    print(f"{batch}/{l}")
    input_image = doct['input_image'].to(device)
    input_predicted_keypoints = doct['input_predicted_keypoints'] \
        .to(device)
    input_predicted_keypoints_valid = doct['input_predicted_keypoints_valid'] \
        .to(device)

    gt_xy = doct['gt_xy'].to(device)
    gt_depth = doct['gt_depth'].to(device)

    use_forearm = doct['use_forearm']

    attention_xy = doct['attention_xy'].to(device)
    attention_depth = doct['attention_depth'].to(device)

    use_forearm = doct['use_forearm'].to(device)
    forearm_direction_gt = doct['forearm_direction'].to(device)

    # tired moses says no pose predictions, don't want degenerate model
    input_predicted_keypoints = torch.zeros(
        input_predicted_keypoints.shape, dtype=torch.float32)
    model_pred_xy, model_pred_depth, model_pred_forearm_3d_and_conf = model(input_image, torch.flatten(
        input_predicted_keypoints, start_dim=1), torch.zeros(input_predicted_keypoints_valid.shape, dtype=torch.float32))

    loss_hmap = loss_fn(model_pred_xy, gt_xy, attention_xy,
                        model_pred_depth, gt_depth, attention_depth)
    loss_forearm = forearm_loss_fn(
        model_pred_forearm_3d_and_conf, forearm_direction_gt, use_forearm)

    print(float(loss_hmap), float(loss_forearm))
    loss = loss_hmap + loss_forearm

    total_loss += float(loss)
    loss_divisor += 1

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    wandb.log({"loss": float(loss), "loss_hmap": float(
        loss_hmap), "loss_forearm": float(loss_forearm)})

    if (batch < 100) or (batch % 15 == 0):

      display_output(
          input_image[0][0].detach().cpu().numpy(),
          gt_xy[0].detach().cpu().numpy(),
          attention_xy[0].detach().cpu().numpy(),
          model_pred_xy[0].detach().cpu().numpy(),
          forearm_direction_gt[0].detach().cpu().numpy(),
          model_pred_forearm_3d_and_conf[0].detach().cpu().numpy(),
          use_forearm[0].detach().cpu().numpy(),
          gui=gui,
      )

  print(f"Avg loss this epoch: {total_loss/loss_divisor}")
  return total_loss


def main():
  parser = argparse.ArgumentParser(description='Train keypoints network')
  parser.add_argument('--no-gui',
                      help="Don't run the little debug gui",
                      dest='gui', action='store_false'
                      )
  parser.add_argument('--gpu',
                      help="Use GPU",
                      dest='gpu', action='store_true'
                      )
  parser.add_argument('--fast',
                      help="Fast test with low batch size.",
                      dest='fast', action='store_true'
                      )
  parser.set_defaults(gui=True)
  parser.set_defaults(gpu=False)
  parser.set_defaults(fast=False)
  args = parser.parse_args()

  print(args.gui)
  print(args.gpu)

  # Subsequent GPU/CPU stuff is cursed but I will not make it perfect, sorry.
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # CPU defaults
  num_devices = 1
  batch_size_per_device = 32

  if (args.gpu):
    num_devices = torch.cuda.device_count()
    print(f"Let's use {num_devices} GPUs!")
    # Warning, this can OOM your RAM if too high. At least right now. Be careful :)
    batch_size_per_device = 128

  if args.fast:
    batch_size_per_device = 1

    wandb.init(project="heatmap_2", entity="col", name="0", mode="disabled")
  else:
    wandb.init(project="heatmap_2", entity="col", name="0")

  hd_train = AllOfTheDatasetsCombined()

  dataloader_train = DataLoader(hd_train, batch_size=batch_size_per_device*num_devices,
                                shuffle=True, num_workers=1 if args.fast else 10)

  model = KeyNet.KeyNet()

  model = torch.nn.DataParallel(model).to(device)

  optimizer = torch.optim.AdamW(model.module.parameters())

  checkpoint_file = os.path.join(
      "checkpoints", 'checkpoint.pth'
  )

  best_performance = 100000000000000

  if os.path.exists(checkpoint_file):
    checkpoint = torch.load(
        checkpoint_file, map_location=torch.device(device))
    best_performance = checkpoint['best_performance']
    model.module.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

  loss_fn = MosesHeatmapLoss()
  forearm_loss_fn = MosesForearmLoss()

  for epoch in range(0, 2000000000000):
    print(f"Epoch {epoch}\n---------------------------------------")
    total_loss = train_loop(device, dataloader_train,
                            model, optimizer, loss_fn, forearm_loss_fn, gui=args.gui)
    final_output_dir = "checkpoints"
    best_model = False
    if total_loss < best_performance:
      best_model = True
      best_performance = total_loss

    save_checkpoint({
        'epoch': epoch,
        'model': "i dont know what to call this",
        'state_dict': model.module.state_dict(),
        'best_performance': best_performance,  # float
        'optimizer': optimizer.state_dict(),
    }, final_output_dir)

    if epoch % 10 == 0:
      print(f"Epoch {epoch}, saving extra!")
      os.system(
          f"cp checkpoints/checkpoint.pth checkpoints/checkpoint_{epoch}.pth")
    if best_model:
      print("Best! Saving as such!")
      os.system(
          "cp checkpoints/checkpoint.pth checkpoints/checkpoint_best.pth")


if __name__ == "__main__":
  main()


# 9.80GB 6.2% at mar 8 02:22
