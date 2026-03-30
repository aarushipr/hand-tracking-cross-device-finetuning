import os
import logging

import torch
import torch.nn as nn

# import model_cfgs
import KeyNet


def meow():
    batch_size = 1
    inp = (
        torch.randn(
            batch_size, 1, 128, 128, dtype=torch.float32), torch.randn(
            batch_size, 42, dtype=torch.float32), torch.randn(
                batch_size, dtype=torch.float32))
    net = KeyNet.KeyNet()

    print("Small - num parameters:", sum(p.numel()
          for p in net.parameters() if p.requires_grad))

    if False:
        ep = 150

        checkpoint_file = os.path.join(
            "checkpoints", f'checkpoint_{ep}.pth'
        )
    else:
        checkpoint_file = os.path.join(
            "checkpoints", 'checkpoint.pth'
        )
    net.eval()

    net(*inp)

    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(
            checkpoint_file,
            map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint["state_dict"])
        print("loaded state dict")
    else:
        raise

    torch.onnx.export(net,         # model being run
                      inp,       # model input (or a tuple for multiple inputs)
                      "marg.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=[
                          'inputImg',
                          'lastKeypoints',
                          'useLastKeypoints'],
                      # the model's input names
                      output_names=[
                          'heatmap_xy',
                          'heatmap_depth',
                          'scalar_extras',
                          'curls'],
                      # the model's output names
                      verbose=False,
                      # dynamic_axes={'inputImg': {0: 'batch_size'}, 'lastKeypoints': },    # variable length axes
                      #                 'x_axis_hmap': {0: 'batch_size'},
                      #                 'y_axis_hmap': {0: 'batch_size'}}
                      )
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    meow()
