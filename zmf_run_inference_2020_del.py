import argparse
from path import Path

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models,os,glob,rawpy,sys
from tqdm import tqdm

import torchvision.transforms as transforms
import flow_transforms
from imageio import imread, imwrite
import numpy as np
from util import flow2rgb

import cv2 as cv

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch FlowNet inference on a folder of img pairs', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='DIR',
                    default='../../data/dataset/data_all_nikon_half/data_all_nikon_half/', help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH',
                    default='./flying_chairs/ft_DFC_VBOF/flownets,adam,300epochs,b32,lr0.0001/model_best.pth.tar', help='path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR',
                    default='./model_test/flownets/ft_dfc_vbof/', help='path to output folder. If not set, will be created in data folder')


parser.add_argument('--output-value', '-v', choices=['raw', 'vis', 'both'], default='both', help='which value to output, between raw input (as a npy file) and color vizualisation (as an image file).' ' If not set, will output both')
parser.add_argument('--div-flow', default=20, type=float, help='value by which flow will be divided. overwritten if stored in pretrained file')
parser.add_argument("--img-exts", metavar='EXT', default=['png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument('--max_flow', default=None, type=float, help='max flow value. Flow map color is saturated above this value. If not set, will use flow map\'s max value')
parser.add_argument('--upsampling', '-u', choices=['nearest', 'bilinear'], default='bilinear', help='if not set, will output FlowNet raw input,' 'which is 4 times downsampled. If set, will output full resolution flow map, with selected upsampling')
parser.add_argument('--bidirectional', action='store_true', help='if set, will output invert flow (from 1 to 0) along with regular flow')

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
     print("wrong: use cpu")


TAG_FLOAT = 202021.25
def flow_write(flow, dst_file):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    """
    # Create the output folder, if necessary
    # Empty the output folder of previous predictions, if any

    # Save optical flow to disk
    with open(dst_file, 'wb') as f:
        np.array(TAG_FLOAT, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)

@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    save_path = args.output
    print('=> will save everything to {}'.format(save_path))

    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
        transforms.Normalize(mean=[0.411,0.432,0.45], std=[1,1,1])
    ])

    img_pairs = []
    names = []
    for ext in args.img_exts:
        test_files = data_dir.files('*1.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.namebase[:-1] + '2.{}'.format(ext))
            if img_pair.isfile():
                img_pairs.append([file, img_pair])
                names.append(file.namebase[:-1])
    # print(img_pairs[:20])
    # print(names[:10])

    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    if 'div_flow' in network_data.keys():
        args.div_flow = network_data['div_flow']

    count = -1
    for (img1_file, img2_file) in tqdm(img_pairs):

        img1 = input_transform(imread(img1_file))
        img2 = input_transform(imread(img2_file))

        input_var = torch.cat([img1, img2]).unsqueeze(0)

        # if args.bidirectional:
        #     # feed inverted pair along with normal pair
        #     inverted_input_var = torch.cat([img2, img1]).unsqueeze(0)
        #     input_var = torch.cat([input_var, inverted_input_var])

        input_var = input_var.to(device)
        # compute output
        output = model(input_var)
        if args.upsampling is not None:
            output = F.interpolate(output, size=img1.size()[-2:], mode=args.upsampling, align_corners=False)
        
        count += 1
        for suffix, flow_output in zip(['flow', 'inv_flow'], output):
            filename = '%s/%s'%(save_path,names[count])
            # print(filename)
            # rgb_flow = flow2rgb(flow_output, max_value=args.max_flow)
            # to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
            # imwrite(filename + '.png', to_save)
            to_save = (flow_output).cpu().numpy().transpose(1,2,0)
            flow_write(to_save, filename+'.flo')

if __name__ == '__main__':
    main()
