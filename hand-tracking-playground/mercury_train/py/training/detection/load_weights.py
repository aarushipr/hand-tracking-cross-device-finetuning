import os
import onnx
from onnx import numpy_helper
import torch

import DetNet

ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "..", "hand-tracking-models", "grayscale_detection_160x160.onnx")

def _load_conv_bn(conv, bn, w_init, b_init):
    weight = torch.from_numpy(numpy_helper.to_array(w_init).copy())
    bias = torch.from_numpy(numpy_helper.to_array(b_init).copy())
    
    with torch.no_grad():
        conv.weight.copy_(weight)
        if conv.bias is not None:
            conv.bias.copy_(bias)
        else:
            conv.bias = torch.nn.Parameter(bias)

    bn.eps = 0.0
    
def load_detnet_weights(model, onnx_path=ONNX_PATH):
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
               
    pairs = list(zip(fused_tensors[0::2], fused_tensors[1::2]))
    
    pair_iter = iter(pairs)
    
    backbone_children = list(model.backbone.children())
    stem_conv = backbone_children[0]
    stem_bn = backbone_children[1]
    
    _load_conv_bn(stem_conv, stem_bn, *next(pair_iter))
    
    for block in backbone_children[3:]:  # skip stem conv/bn/relu, indices 0,1,2
        irb_layers = list(block.conv.children())
        conv_bn_layers = [m for m in irb_layers if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d))]

        for i in range(0, len(conv_bn_layers), 2):
            conv = conv_bn_layers[i]
            bn = conv_bn_layers[i + 1]
            _load_conv_bn(conv, bn, *next(pair_iter))

    remaining = list(pair_iter)
    assert len(remaining) == 0, f"{len(remaining)} pairs left unconsumed!"

    return model

if __name__ == "__main__":
    model = DetNet.DetNet()
    load_detnet_weights(model)
    