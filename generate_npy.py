import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from raft_stereo import RAFTStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import pdb


DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_mask(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).float()
    return img[None].to(DEVICE)

def demo(args):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))

        # if a mask is given
        if args.left_mask_imgs is not None and args.right_mask_imgs is not None:
            '''
            python generate_npy.py --restore_ckpt models/raftstereo-realtime.pth --shared_backbone \
                --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg \
                    --mixed_precision -l=/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/*.png \
                        -r=/media/emmah/PortableSSD/Arclab_data/trial_9_right_rgb/*.png --save_numpy \
                            --output_directory=/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output_mask \
                                --left_mask_imgs=/media/emmah/PortableSSD/Arclab_data/trial_9_single_arm_no_tension_masks_meat_left/trial_9_single_arm_no_tension_masks_meat/*.png \
                                    --right_mask_imgs=/media/emmah/PortableSSD/Arclab_data/trial_9_single_arm_no_tension_masks_meat_right/trial_9_single_arm_no_tension_masks_meat/*.png
            '''
            left_mask_images = sorted(glob.glob(args.left_mask_imgs, recursive=True))
            right_mask_images = sorted(glob.glob(args.right_mask_imgs, recursive=True))

            print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

            for (imfile1, imfile2, left_mask, right_mask) in tqdm(list(\
                zip(left_images, right_images, left_mask_images, right_mask_images))):
                # pdb.set_trace()
                l_image = load_image(imfile1)
                r_image = load_image(imfile2)
                left_mask = load_mask(left_mask)
                right_mask = load_mask(right_mask)

                left_mask = left_mask % 224
                right_mask = right_mask % 224

                l_image = l_image * left_mask
                r_image = r_image * right_mask



                padder = InputPadder(l_image.shape, divis_by=32)
                l_image, r_image = padder.pad(l_image, r_image)

                _, flow_up = model(l_image, r_image, iters=args.valid_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).squeeze()

                file_stem = (imfile1.split('/')[-1]).split('.')[-2]
                if args.save_numpy:
                    np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
                plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')


        # if no mask is given
        else:
            '''python generate_npy.py --restore_ckpt models/raftstereo-realtime.pth \
                --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 \
                    --corr_implementation reg --mixed_precision -l=/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/*.png \
                        -r=/media/emmah/PortableSSD/Arclab_data/trial_9_right_rgb/*.png --save_numpy \
                            --output_directory=/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output_no_mask
            '''            
            print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

            for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
                l_image = load_image(imfile1)
                r_image = load_image(imfile2)

                padder = InputPadder(l_image.shape, divis_by=32)
                l_image, r_image = padder.pad(l_image, r_image)

                _, flow_up = model(l_image, r_image, iters=args.valid_iters, test_mode=True)
                flow_up = padder.unpad(flow_up).squeeze()

                file_stem = (imfile1.split('/')[-1]).split('.')[-2]
                if args.save_numpy:
                    np.save(output_directory / f"{file_stem}.npy", flow_up.cpu().numpy().squeeze())
                plt.imsave(output_directory / f"{file_stem}.png", -flow_up.cpu().numpy().squeeze(), cmap='jet')

''' python demo.py --restore_ckpt models/raftstereo-realtime.pth --shared_backbone --n_downsample 3 \
--n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg --mixed_precision \
-l=/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/*.png -r=/media/emmah/PortableSSD/Arclab_data/trial_9_right_rgb/*.png \
--save_numpy --output_directory=/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output


'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/*.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="/media/emmah/PortableSSD/Arclab_data/trial_9_right_rgb/*.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--left_mask_imgs', help='path to all left mask images', default=None)
    parser.add_argument('--right_mask_imgs', help='path to all right mask images', default=None)
    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)
