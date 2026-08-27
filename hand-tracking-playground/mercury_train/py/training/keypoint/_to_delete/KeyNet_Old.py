import torch
import torch.nn as nn
from InvertedResidual import InvertedResidual

num_input_joints = 21
num_output_heatmap_joints = 21

# existence (1) + forearm (3) + GNLL curls (5*2)
# 14. so, to be safe, let's do 20 extras


class KeyNet(nn.Module):
    def __init__(self, input_side_px=128, num_extras=8):
        super(KeyNet, self).__init__()

        # Needs to be divisible by 8!
        assert (input_side_px % 8) == 0
        self.input_side_px = input_side_px

        # This is complicated, it's not the output though. Not totally sure -
        # Moses, may 5.
        self.input_predicted_keypoints_side_px = int(input_side_px / 8)

        if input_side_px == 128:
            self.output_hmap_side = 22
        elif input_side_px == 96:
            self.output_hmap_side = 18
        else:
            print("This won't work right! Read the code nerd")
            self.output_hmap_side = 22

        self.image_network = self.make_backbone_image()
        self.keypoints_network = self.make_backbone_keypoints()
        self.fused_network = self.make_backbone_fused()
        self.network_1d_depth = self.depth_regression_1d()
        self.network_extras = self.extras_regression(num_extras)
        self.network_curls = self.curls_regression()
        self.network_2d_px_coord = self.px_coord_regression_2d()

        self.init_weights()

    def depth_regression_1d(self):
        out = nn.Sequential(
            nn.AvgPool2d(kernel_size=6, stride=6),

            # 21*18 = 378

            nn.Conv2d(in_channels=160, out_channels=self.output_hmap_side * \
                      num_output_heatmap_joints, kernel_size=1),
            nn.ReLU6(inplace=True),
        )
        return out

    def extras_regression(self, num_extras):
        in_size = 640 * 1  # randomly guessed from netron
        intermediate_size = 128
        # in_size = 69
        out = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Flatten(),
            nn.Linear(in_size, 128),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU6(inplace=True),
            nn.Linear(intermediate_size, num_extras),

            # 21*18 = 378

            # nn.Conv2d(in_channels=160, out_channels=16, kernel_size=1),
        )
        return out

    def curls_regression(self):
        in_size = 640 * 1  # randomly guessed from netron
        intermediate_size = 128
        # in_size = 69
        out = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=3),
            nn.Flatten(),
            nn.Linear(in_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU6(inplace=True),
            nn.Linear(intermediate_size, 10),
            # nn.ReLU6(inplace=True), # Stops the outputs from being below 0

            # 21*18 = 378

            # nn.Conv2d(in_channels=160, out_channels=16, kernel_size=1),
        )
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
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

    def make_backbone_image(self):

        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 32, 1, 1],
            [6, 32, 1, 2],
            [6, 32, 1, 1],
            [6, 64, 1, 2],
            [6, 64, 2, 1],
        ]

        out = [
            nn.Conv2d(1, input_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        ]

        for t, c, n, s in interverted_residual_setting:

            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(
                        input_channel, output_channel, s, expand_ratio=t))
                else:
                    out.append(InvertedResidual(
                        input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        return nn.Sequential(*out)

    def make_backbone_keypoints(self):
        depth = 32
        print(num_input_joints * 3)
        out = nn.Sequential(nn.Linear(num_input_joints *
                                      3, int(self.input_predicted_keypoints_side_px *
                                             self.input_predicted_keypoints_side_px *
                                             depth)), nn.ReLU6(inplace=True))
        return out

    def make_backbone_fused(self, input_channel=96):
        interverted_residual_setting = [
            # t, c, n, s
            [1, 64, 2, 1],
            [1, 64, 3, 1],
            [1, 96, 1, 1],
            [1, 96, 2, 1],
            [1, 128, 1, 2],
            [1, 128, 2, 1],
            [1, 160, 1, 1],
        ]

        out = []
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    out.append(InvertedResidual(
                        input_channel, output_channel, s, expand_ratio=t))
                else:
                    out.append(InvertedResidual(
                        input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        return nn.Sequential(*out)

    def px_coord_regression_2d(self):
        out = nn.Sequential(
            nn.Conv2d(
                in_channels=160,
                out_channels=num_output_heatmap_joints * 3,
                kernel_size=3,
                padding=2),
            nn.BatchNorm2d(
                num_output_heatmap_joints * 3),
            nn.ReLU6(
                inplace=True),
            nn.ConvTranspose2d(
                in_channels=num_output_heatmap_joints * 3,
                out_channels=num_output_heatmap_joints * 2,
                kernel_size=2,
                stride=2),
            nn.Conv2d(
                in_channels=num_output_heatmap_joints * 2,
                out_channels=num_output_heatmap_joints,
                kernel_size=3,
                padding=2),
            nn.BatchNorm2d(num_output_heatmap_joints),
            nn.ReLU6(
                inplace=True))
        return out

    def forward(
            self,
            x,
            addon=torch.ones(
                1,
                num_output_heatmap_joints *
                3),
            use_addon=1.):
        x = self.image_network(x)
        x_addon = self.keypoints_network(addon)  # b, 4608

        x_addon = x_addon.view(-1, 32, self.input_predicted_keypoints_side_px,
                               self.input_predicted_keypoints_side_px)

        # https://discuss.pytorch.org/t/how-to-perform-an-scalar-element-wise-multiplication/29253
        x_addon = use_addon[:, None, None, None] * x_addon

        x = torch.cat((x, x_addon), dim=1)  # b, 96, 12, 12
        x = self.fused_network(x)
        out_dist_hmap_1d = self.network_1d_depth(x)
        out_dist_hmap_1d = torch.reshape(
            out_dist_hmap_1d, (-1, num_output_heatmap_joints, self.output_hmap_side))

        out_extras = self.network_extras(x)

        out_px_coord_hmap_2d = self.network_2d_px_coord(x)
        out_curls = self.network_curls(x)

        return out_px_coord_hmap_2d, out_dist_hmap_1d, out_extras, out_curls


if __name__ == "__main__":
    model = KeyNet()
    batch_size = 1
    inp = (
        torch.randn(
            batch_size, 1, 128, 128, dtype=torch.float32), torch.randn(
            batch_size, 63, dtype=torch.float32), torch.randn(
                batch_size, dtype=torch.float32))
    a = model(*inp)
    print(a[2])
