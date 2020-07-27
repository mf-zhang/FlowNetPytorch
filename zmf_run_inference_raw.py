import argparse

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


parser.add_argument('--data', metavar='DIR',default='../../data/canon', help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', default='../../data/flownetc_EPE1.766.tar', help='path to pre-trained model')
# parser.add_argument('--pretrained', metavar='PTH',default='./flying_chairs/10-24-10:14/flownetc,adam,300epochs,b64,lr0.0001/model_best.pth.tar', help='path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR', default='./c-canon-nlm-au', help='path to output folder. If not set, will be created in data folder')


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

def crop(im,h,w,sample_every_n_pixels):
    h0 = im.shape[0]
    w0 = im.shape[1]
    if (h0 < h or w0 < w):
        print("bad crop")
        return im
    newim = im[0:h:sample_every_n_pixels,0:w:sample_every_n_pixels,:]
    return newim

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

    data_dir = args.data
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
    # for ext in args.img_exts:
    #     test_files = data_dir.files('*1.{}'.format(ext))
    #     for file in test_files:
    #         img_pair = file.parent / (file.namebase[:-1] + '2.{}'.format(ext))
    #         if img_pair.isfile():
    #             img_pairs.append([file, img_pair])
    #             print(img_pairs)
    canon_im1num = [1,1,2,10,10,11,21,21,21,21,22,22,22,24,25,31,31,31,32,32,33,41,43,51,54] # 25
    canon_im2num = [2,3,3,11,12,12,22,24,25,26,23,24,27,26,27,32,33,34,33,34,34,42,45,52,55]
    # im1 = '%s/1/(1).JPG'%data_dir
    # im2 = '%s/2/(1).JPG'%data_dir
    # img_pairs.append([im1,im2])

    s = 0
    for i in range(0,25):
        foldernum1 = canon_im1num[i]
        foldernum2 = canon_im2num[i]
        rawnum = len(glob.glob('../../data/canon/%d/*.CR2'%(foldernum1)))
        for j in range(rawnum):
            image_path1 = '../../data/canon/%d/(%d).CR2'%(foldernum1,j+1)
            image_path2 = '../../data/canon/%d/(%d).CR2'%(foldernum2,j+1)

            img_pairs.append([image_path1, image_path2])
            n = '%d-%d_%d'%(foldernum1,foldernum2,j+1)
            names.append(n)


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

        raw1 = rawpy.imread(img1_file)
        raw2 = rawpy.imread(img2_file)

        im1 = raw1.raw_image_visible.astype(np.float32)
        im1 = (im1 - 2047) / (16383 - 2047)
        ratio = 0.3 / np.mean(im1)
        im1 = np.minimum(np.maximum(im1*ratio,0.0),1.0)

        im2 = raw2.raw_image_visible.astype(np.float32)
        im2 = (im2 - 2047) / (16383 - 2047)
        ratio = 0.3 / np.mean(im2)
        im2 = np.minimum(np.maximum(im2*ratio,0.0),1.0)
        # print(ratio)

        im1 = np.expand_dims(im1, axis=2)
        H = im1.shape[0]
        W = im1.shape[1]
        image1 = np.concatenate((im1[0:H:2, 0:W:2,:], #r
                            (im1[0:H:2, 1:W:2,:]+im1[1:H:2, 0:W:2,:])/2.0, #g
                            im1[1:H:2, 1:W:2,:]), axis=2) #b
        im2 = np.expand_dims(im2, axis=2)
        H = im2.shape[0]
        W = im2.shape[1]
        image2 = np.concatenate((im2[0:H:2, 0:W:2,:], #r
                            (im2[0:H:2, 1:W:2,:]+im2[1:H:2, 0:W:2,:])/2.0, #g
                            im2[1:H:2, 1:W:2,:]), axis=2) #b
        i1 = crop(image1,1920,2944,2)
        i2 = crop(image2,1920,2944,2)

        i1 = (i1*255).astype('uint8')
        i2 = (i2*255).astype('uint8')

        i1 = cv.fastNlMeansDenoisingColored(i1,None,10,10,7,21)
        i2 = cv.fastNlMeansDenoisingColored(i2,None,10,10,7,21)

        img1 = input_transform(i1)
        img2 = input_transform(i2)

        # print(i1.shape)
        # print(img1.shape)

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
            print(filename)
            # if args.output_value in['vis', 'both']:
            rgb_flow = flow2rgb(flow_output, max_value=args.max_flow)
            to_save = (rgb_flow * 255).astype(np.uint8).transpose(1,2,0)
            imwrite(filename + '.png', to_save)
            # if args.output_value in ['raw', 'both']:
            #     # Make the flow map a HxWx2 array as in .flo files
            to_save = (flow_output).cpu().numpy().transpose(1,2,0)
            # print(to_save.shape)
            flow_write(to_save, filename+'.flo')

            #     np.save(filename + '.npy', to_save)


if __name__ == '__main__':
    main()
