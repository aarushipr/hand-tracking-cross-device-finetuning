import os
import onnx
from onnx import numpy_helper
import torch

import KeyNet

ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "hand-tracking-models", "grayscale_keypoint_jan18.onnx")


def load_keynet_weights(model, onnx_path=ONNX_PATH):
    onnx_model = onnx.load(onnx_path)

    named_tensors = {}
    fused_tensors = []

    for raw_tensor in onnx_model.graph.initializer:
        if raw_tensor.name.startswith("onnx::"):
            fused_tensors.append(raw_tensor)
        else:
            named_tensors[raw_tensor.name] = numpy_helper.to_array(raw_tensor)

    state_dict = model.state_dict()
    for name, array in named_tensors.items():
        state_dict[name].copy_(torch.from_numpy(array))

    # pair the 1st and 2nd element
    pairs = list(zip(fused_tensors[0::2], fused_tensors[1::2]))
    pair_iter = iter(pairs)
    
    # .children() returns direct submodules
    image_children = list(model.image_network.children())
    stem_conv = image_children[0]
    stem_bn = image_children[1]

    load_conv_bn(stem_conv, stem_bn, *next(pair_iter))

    for block in image_children[3:]:
        irb_layers = list(block.conv.children())
        conv_bn_layers = [m for m in irb_layers if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]
        
        for i in range(0, len(conv_bn_layers), 2):
            conv = conv_bn_layers[i]
            bn = conv_bn_layers[i + 1]
            load_conv_bn(conv, bn, *next(pair_iter))

    for block in model.fused_network.children():
        irb_layers = list(block.conv.children())
        conv_bn_layers = [m for m in irb_layers if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]

        for i in range(0, len(conv_bn_layers), 2):
            conv = conv_bn_layers[i]
            bn = conv_bn_layers[i + 1]
            load_conv_bn(conv, bn, *next(pair_iter))
        
    load_conv_bn(model.network_2d_px_coord[0], model.network_2d_px_coord[1], *next(pair_iter))
    load_conv_bn(model.network_2d_px_coord[4], model.network_2d_px_coord[5], *next(pair_iter))

    remaining = list(pair_iter)
    assert len(remaining) == 0, f"{len(remaining)} pairs left unconsumed!"

    return model


def load_conv_bn(conv, bn, w_init, b_init):
    # convert onnx format to pytorch tensor
    weight = torch.tensor(numpy_helper.to_array(w_init))
    bias = torch.tensor(numpy_helper.to_array(b_init))

    # no_grad() because this is not a training step
    with torch.no_grad():
        conv.weight.copy_(weight)
        if conv.bias is not None:
            conv.bias.copy_(bias)
        else:
            conv.bias = torch.nn.Parameter(bias)

        bn.eps = 0.0
   
        
if __name__ == "__main__":
    import numpy as np

    model = KeyNet.KeyNet()
    model.eval()
    print("BEFORE image:", model.image_network[8].conv[6].weight[0, 0, :5])
    print("BEFORE fused:", model.fused_network[11].conv[3].weight[0, 0, :5])
    load_keynet_weights(model)
    print("AFTER image:", model.image_network[8].conv[6].weight[0, 0, :5])
    print("AFTER fused:", model.fused_network[11].conv[3].weight[0, 0, :5])

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        torch.manual_seed(0)
        x = torch.randn(1, 1, 128, 128)
        kp = torch.randn(1, 42)
        valid = torch.ones(1)
        with torch.no_grad():
            xy_t, depth_t, extras_t, curls_t = model(x, kp, valid)
        xy_o, depth_o, extras_o, curls_o = sess.run(
            None, {"inputImg": x.numpy(), "lastKeypoints": kp.numpy(),
                   "useLastKeypoints": valid.numpy()})
        for name, t, o in [("heatmap_xy", xy_t, xy_o), ("heatmap_depth", depth_t, depth_o),
                            ("scalar_extras", extras_t, extras_o), ("curls", curls_t, curls_o)]:
            print(f"{name} max abs diff vs ONNX: {np.max(np.abs(t.numpy() - o)):.2e}")
    except ImportError:
        print("(onnxruntime not installed -- skipping numerical cross-check)")