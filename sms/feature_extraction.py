# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    parser.add_argument('--data-prefix', default='data/ActivityNet/rawframes', help='dataset prefix')
    parser.add_argument('--output-prefix', default='data/ActivityNet/rgb_feat', help='output prefix')
    parser.add_argument(
        '--data-list', default='data/ActivityNet/anet_train_video.txt',
        help='video list of the dataset, the format should be '
        '`frame_dir num_frames output_file`')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=1,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--ckpt', default='checkpoints/tsn_r50_320p_1x1x8_50e_activitynet_video_rgb_20210301-7f8da0c6.pth', help='checkpoint for feature extraction')
    parser.add_argument(
        '--start', type=int, default=0, help='start index')
    parser.add_argument(
        '--end', type=int, default=-1, help='end index')
    parser.add_argument(
        '--model_struct', choices=['r50', 'r101', 'r152'], default='r50', help='feature model structure')
    parser.add_argument(
        '--dataset', choices=['ActivityNet', 'FCVID', 'UCF101'], default='ActivityNet', help='video dataset')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.is_rgb = args.modality == 'RGB'
    args.clip_len = 1 if args.is_rgb else 5
    args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
    rgb_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)
    # max batch_size for one forward
    args.batch_size = 500

    # define the data pipeline for Untrimmed Videos
    data_pipeline = [
        dict(
            type='UntrimmedSampleFrames',
            clip_len=args.clip_len,
            frame_interval=args.frame_interval,
            start_index=0),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=256),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format=args.input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    if args.dataset == 'ActivityNet':
        num_classes = 200
    elif args.dataset == 'FCVID':
        num_classes = 239
    elif args.dataset == 'UCF101':
        num_classes = 101

    model_depth = int(args.model_struct[1:])
    # define TSN R50 model, the model is used as the feature extractor
    model_cfg = dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNet',
            depth=model_depth,
            in_channels=args.in_channels,
            norm_eval=False),
        cls_head=dict(
            type='TSNHead',
            num_classes=num_classes,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1)),
        test_cfg=dict(average_clips=None, feature_extraction=True))
    model = build_model(model_cfg)
    # load pretrained weight into the feature extractor
    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict)
    model = model#.cuda()
    model.eval()

    data = open(args.data_list).readlines()
    data = [x.strip() for x in data]

    if args.end < 0:
        data = data[args.start:]
    else:
        data = data[args.start:args.end]

    print(f'to extract {len(data)} videos')

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(args.output_prefix):
        os.system(f'mkdir -p {args.output_prefix}')

    mean_feature_out = {}

    for item in data:
        frame_dir, length, label = item.split()
        print(frame_dir)
        output_file = osp.basename(frame_dir) + '_r50.pkl'# f'_{args.model_struct}.pkl'
        frame_dir = osp.join(args.data_prefix, frame_dir)
        output_file = osp.join(args.output_prefix, output_file)
        assert output_file.endswith('.pkl')
        length = int(length)

        # prepare a pseudo sample
        tmpl = dict(
            frame_dir=frame_dir,
            total_frames=length,
            filename_tmpl=args.f_tmpl,
            start_index=1,
            modality=args.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs']
        shape = imgs.shape
        # the original shape should be N_seg * C * H * W, resize it to N_seg *
        # 1 * C * H * W so that the network return feature of each frame (No
        # score average among segments)
        imgs = imgs.reshape((shape[0], 1) + shape[1:])
        imgs = imgs#.cuda()


        def forward_data(model, data, label):
            # chop large data into pieces and extract feature from them
            features = []
            start_idx = 0
            num_clip = imgs.shape[0]
            with torch.no_grad():
                part = imgs
                label = torch.LongTensor([int(label)])#.cuda()
                gt_prob, feat = model.forward_train_features(part, label)
                features.append(feat.cpu().numpy())
            res = {
                'features': np.concatenate(features),
                'gt_prob': gt_prob.cpu().numpy()
            }
            return res

        res = forward_data(model, imgs, label)

        mean_feature_out[osp.basename(frame_dir)] = res['features'].mean(axis=0)

        with open(output_file, 'wb') as fout:
            pickle.dump(res, fout)
        
        prog_bar.update()



if __name__ == '__main__':
    main()
