import argparse
from pathlib import Path
import pickle
import torch
import tensorrt as trt
from torch2trt import torch2trt, tensorrt_converter
from torch2trt.torch2trt import add_missing_trt_tensors
from TCN.mnist_pixel.model import TCN

parser = argparse.ArgumentParser(description='Conversion parameters')
parser.add_argument('--mdlpath', type=Path, default='./TCN/mnist_pixel/models',
    help='Path to torch model folder')
parser.add_argument('--mdlname', type=str,
    help='Name of the torch model to convert')
parser.add_argument('--trtpath', type=Path, default='./TCN/mnist_pixel/models_trt',
    help='Path to TensorRT model folder')
parser.add_argument('--onnx', action='store_true',
    help='Convert to ONNX before converting to TensorRT')
parser.add_argument('--applymax', action='store_true',
    help='Apply argmax within the TensorRT engine')
parser.add_argument('--fp16', action='store_true',
    help='Use half precision floats')

# python convert_trt.py --mdlname aug_k7l6 --applymax
# python convert_trt.py --mdlname aug_k7l6 --applymax --fp16
# python convert_trt.py --mdlname aug_k7l6 --applymax --onnx
# python convert_trt.py --mdlname aug_k7l6 --applymax --onnx --fp16

# Converter for argmax wrapper
@tensorrt_converter('TCN.mnist_pixel.model.trt_argmax')
def convert_argmax(ctx):
    input_tensor = ctx.method_args[0]
    # The K in "topk"
    axis = ctx.method_kwargs['dim']
    output = ctx.method_return
    
    if axis is None:
        input_tensor = torch.flatten(input_tensor)
        axis = 0
    
    input_tensor_trt = add_missing_trt_tensors(ctx.network, [input_tensor])[0]

    layer = ctx.network.add_topk(input_tensor_trt, trt.TopKOperation.MAX, 1, axis)
    
    # layer.get_output(0) would give the max value
    output._trt = layer.get_output(1)

if __name__ == '__main__':
    batch_size, in_channels, n_classes = 1, 1, 10
    args = parser.parse_args()

    mdl_state_path = args.mdlpath / (args.mdlname + '.pt')
    mdl_param_path = args.mdlpath / (args.mdlname+'_args.pkl')
    mdl_params = pickle.load(open(mdl_param_path, 'rb'))

    channel_sizes = [mdl_params.nhid] * mdl_params.levels

    model = TCN(in_channels, 
                n_classes, 
                channel_sizes, 
                kernel_size=mdl_params.ksize,
                trt=not args.onnx,
                apply_max=args.applymax
                )
    
    model.load_state_dict(torch.load(mdl_state_path), strict=False)
    model.eval()
    model.cuda()
    print('Successfully loaded torch model')

    x = torch.rand((1,1,28*28)).cuda()

    if args.fp16:
        model = model.half()
        x = x.half()

    model_trt = torch2trt(model, [x], strict_type_constraints=True, use_onnx=args.onnx, fp16=args.fp16)
    print('Successfully converted model to TensorRT engine')

    mdl_trt_name = args.mdlname + '_trt'
    mdl_trt_name = mdl_trt_name + '_onnx' if args.onnx else mdl_trt_name
    mdl_trt_name = mdl_trt_name + '_amax' if args.applymax else mdl_trt_name
    mdl_trt_name = mdl_trt_name + '_fp16' if args.fp16 else mdl_trt_name

    print(f'Output Directory: \n {args.trtpath}')
    torch.save(model_trt.state_dict(), args.trtpath / (mdl_trt_name+'.pt'))
    print(f'Saved TRTModule state dict in {mdl_trt_name}.pt')

    with open(args.trtpath / (mdl_trt_name+'.engine'), 'wb') as f:
        f.write(model_trt.engine.serialize())
    print(f'Saved serialized TensorRT engine in {mdl_trt_name}.engine')

