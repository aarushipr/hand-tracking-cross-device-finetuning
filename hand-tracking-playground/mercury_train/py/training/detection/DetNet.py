
import torch
import torch.nn as nn
from py.training.common.irb import InvertedResidual

# Cargo culted from https://github.com/pytorch/pytorch/issues/42653#issuecomment-1168816422
# Didn't work, don't remember why. Worth investigating further.
# class AdaptiveAvgPool2dCustom(nn.Module):
#     def __init__(self, output_size):
#         super(AdaptiveAvgPool2dCustom, self).__init__()
#         self.output_size = np.array(output_size)

#     def forward(self, x: torch.Tensor):
#         stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
#         kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
#         avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
#         x = avg(x)
#         return x


class SizeProbe(nn.Module):
    def __init__(self, name=""):
        super(SizeProbe, self).__init__()
        self.name = name

    def forward(self, x: torch.Tensor):
        print(f"{self.name}:", x.shape)
        return x


class DetNet(nn.Module):
    def __init__(self):
        super(DetNet, self).__init__()

        self.backbone = self.make_backbone()
        self.fc = self.make_fc()
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weights()  # Will get overwritten later by checkpoint loading

    def make_backbone(self):
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 2],
            [6, 32, 1, 2],
            [6, 32, 1, 1],
            [6, 64, 1, 2],
            [6, 64, 2, 1],
            [6, 64, 1, 2],
            [6, 64, 3, 1],
            [6, 96, 1, 1],
            [6, 96, 2, 1],
            [6, 128, 1, 2],
            [6, 128, 2, 1],
            [6, 160, 1, 1],
        ]

        out = [
            # nn.AvgPool2d(kernel_size=4, stride=4),
            nn.Conv2d(1, input_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]

        for t, c, n, s in interverted_residual_setting:

            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(input_channel,
                               output_channel, s, expand_ratio=t))
                else:
                    out.append(InvertedResidual(input_channel,
                               output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        return nn.Sequential(*out)

    # Note: This might be a pretty wrong way of doing it.
    # The original implementation used a Conv1d then an AdaptiveAvgPool2d(2) or so for each output
    # whereas this uses some much bigger fully-connected layers to achieve a similar effect.
    #
    # There's been a pretty big move away from huge Linear/FC layers because they're too big and train slow, and that probably applies here.
    # If you're bored, please try replacing this function with something more sensible, and prove that it works just as well on the validation sets!
    #
    # Also: might there be a better way to shrink the initial 3200-wide tensor?

    def make_fc(self):
        batchnorm = True
        dropout = False
        out = [nn.Flatten()]

        # out.append(nn.Linear(3200, 256)) # 320x240
        # out.append(nn.Linear(2560, 256)) # 224x224
        out.append(nn.Linear(1440, 256)) # 160x160
        # out.append(nn.Linear(960, 256)) # 160x120
        if dropout:
            out.append(nn.Dropout(0.1, inplace=True))
        if batchnorm:
            out.append(nn.BatchNorm1d(256))
        out.append(nn.ReLU6(inplace=True))
        ##

        ##
        out.append(nn.Linear(256, 8))
        if dropout:
            out.append(nn.Dropout(0.1, inplace=True))
        if batchnorm:
            out.append(nn.BatchNorm1d(8))
        out.append(nn.ReLU6(inplace=True))
        ##

        out.append(nn.Linear(8, 8))

        return nn.Sequential(*out)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(
                    m.weight, gain=nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # so onnx export works good
    def onnx_forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        return x

    # so it's easy to train
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)

        # reminder: first dimension is just the batch, so we don't want to slice that one
        # then we're just slicing across the second dimension
        exists = self.sigmoid(x[:, 0:2])
        center_x = x[:, 2:4]
        center_y = x[:, 4:6]
        size = x[:, 6:8]

        return exists, center_x, center_y, size


if __name__ == "__main__":
    bobob = DetNet()
    bobob.eval()

    # bobob.forward = bobob.onnx_forward

    inp = torch.randn(1, 1, 160, 160)

    e = bobob.forward(inp)

    for j in e:
        print(j.shape)

    torch.onnx.export(bobob,         # model being run
                      inp,       # model input (or a tuple for multiple inputs)
                      "blerg.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['inputImg'],   # the model's input names
                      # the model's output names
                      output_names=['hand_visible', 'cx', 'cy', 'size'],
                      verbose=False,
                      # dynamic_axes={'inputImg': {0: 'batch_size'}, 'lastKeypoints': },    # variable length axes
                      #                 'x_axis_hmap': {0: 'batch_size'},
                      #                 'y_axis_hmap': {0: 'batch_size'}}
                      )
    print(" ")
    print('Model has been converted to ONNX')
