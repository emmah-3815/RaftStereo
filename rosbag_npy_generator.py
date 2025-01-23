import os
import argparse
import pdb
import cv2
import numpy as np

import rosbag   
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

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

'''
python rosbag_npy_generator.py --bag_file /media/emmah/PortableSSD/Arclab_data/11_18_trial_001.bag --image_dir /media/emmah/PortableSSD/Arclab_data/meat_thread_data_11_17/trial_001/ --topic_l /stereo/left/image --topic_r /stereo/right/image --output_name trial_001 --restore_ckpt /media/emmah/PortableSSD/Arclab_data/models/raftstereo-realtime.pth --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg --mixed_precision --save_numpy

 python rosbag_npy_generator.py --bag_file /media/emmah/PortableSSD/Arclab_data/11_18_trial_012.bag --image_dir /media/emmah/PortableSSD/Arclab_data/meat_thread_data_11_17/trial_12/ --topic_l /stereo/left/image --topic_r /stereo/right/image --restore_ckpt /media/emmah/PortableSSD/Arclab_data/models/raftstereo-realtime.pth --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 15 --corr_implementation reg --mixed_precision --save_numpy
'''



def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def generate_npy(args, left_img_dir, right_img_dir):
    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.image_dir + "/npy/")
    output_directory.mkdir(exist_ok=True)
    left_imgs = left_img_dir + "*"
    right_imgs = right_img_dir + "*"


    with torch.no_grad():
        # left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        # right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        left_images = sorted(glob.glob(left_imgs, recursive=True))
        right_images = sorted(glob.glob(right_imgs, recursive=True))


        '''python generate_npy.py --restore_ckpt /media/emmah/PortableSSD/Arclab_data/models/raftstereo-realtime.pth \
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

def rosbag_extract(args):

    # args = parser.parse_args()
    # mypath = args.image_dir
    # if not os.path.isdir(mypath):
    #     os.makedirs(mypath)

    left_image_dir = args.image_dir + "left_rgb/"
    if not os.path.isdir(left_image_dir):
        os.makedirs(left_image_dir)
    right_image_dir = args.image_dir + "right_rgb/"
    if not os.path.isdir(right_image_dir):
        os.makedirs(right_image_dir)

    if args.new != True:
        print("using existing images in", left_image_dir, "and", right_image_dir)
        return left_image_dir, right_image_dir

    print ("Extract images from %s on topic %s into %s and topic %s into %s" % (args.bag_file, \
                                                          args.topic_l, left_image_dir, args.topic_r, right_image_dir))

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    # file_path = os.path.join(args.output_dir, args.output_name + "_%06i.npy" % count)
    # points_list = []

    print("extracting left image in topic", args.topic_l)
    for topic, msg, t in bag.read_messages(topics=[args.topic_l]):

        # extract image
        try:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg) # tries compressed image convertion to cv2

        except:
            print("something went wrong")

        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # tries regular image convertion to cv2
        except AttributeError as e:
            try:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg) # tries compressed image convertion to cv2

            except CvBridgeError as e:
                print(e)

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # convert bgr format to rgb
        cv2.imwrite(os.path.join(left_image_dir, args.output_name + "_%06i.png" % count), cv_img)
        print ("Wrote image %i" % count)

        count += 1

    print("extracting right image in topic", args.topic_r)
    count = 0
    for topic, msg, t in bag.read_messages(topics=[args.topic_r]):

        # extract image
        try:
            cv_img = bridge.compressed_imgmsg_to_cv2(msg) # tries compressed image convertion to cv2

        except:
            print("something went wrong")

        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") # tries regular image convertion to cv2
        except AttributeError as e:
            try:
                cv_img = bridge.compressed_imgmsg_to_cv2(msg) # tries compressed image convertion to cv2

            except CvBridgeError as e:
                print(e)

        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # convert bgr format to rgb
        cv2.imwrite(os.path.join(right_image_dir, args.output_name + "_%06i.png" % count), cv_img)
        print ("Wrote image %i" % count)

        count += 1
    # pdb.set_trace()
    ## extract points
    # np.save(file_path, np.array(points_list[0]))
    bag.close()

    return left_image_dir, right_image_dir




if __name__ == '__main__':
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag_file", help="Input ROS bag.")
    parser.add_argument("--image_dir", help="Output directory, (existing or new)")
    parser.add_argument("--new", default=True, help="existing or new images")
    parser.add_argument("--topic_l", help="Left Image topic.")
    parser.add_argument("--topic_r", help="RIght Image topic.")
    parser.add_argument("--output_name", nargs='?', default="frame", help="Image name. (optional)")

    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    # parser.add_argument('--npy_directory', help="directory to save output", default="demo_output")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

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

    DEVICE = 'cpu'

    args = parser.parse_args()

    left_dir, right_dir = rosbag_extract(args)
    generate_npy(args, left_dir, right_dir)


'''
python generate_npy.py --restore_ckpt /media/emmah/PortableSSD/Arclab_data/models/raftstereo-realtime.pth \
            --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 \
                --corr_implementation reg --mixed_precision -l=/media/emmah/PortableSSD/Arclab_data/trial_9_left_rgb/*.png \
                    -r=/media/emmah/PortableSSD/Arclab_data/trial_9_right_rgb/*.png --save_numpy \
                        --output_directory=/media/emmah/PortableSSD/Arclab_data/trial_9_RAFT_output_no_mask
'''
'''
python rosbag_extract.py --bag_file /media/emmah/PortableSSD/Arclab_data/meat_thread_data_9_26/trial_01.bag \
    --output_dir /media/emmah/PortableSSD/Arclab_data/meat_thread_data_9_26/trial_01/ \
        --topic /stereo/left/rectified_downscaled_image --output_name trial_01_left
'''