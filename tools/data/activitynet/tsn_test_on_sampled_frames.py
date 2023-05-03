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
    parser.add_argument('--data-prefix', default='../mmaction2/data/ActivityNet/rawframes', help='dataset prefix')
    parser.add_argument('--feat-prefix', default='../mmaction2/data/ActivityNet/rgb_feat', help='feature prefix')
    parser.add_argument(
        '--data-list', default='../mmaction2/data/ActivityNet/anet_val_video.txt',
        help='video list of the dataset, the format should be '
        '`frame_dir num_frames output_file`')
    parser.add_argument(
        '--frame-interval',
        type=int,
        default=16,
        help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='which part of dataset to forward(alldata[part::total])')
    parser.add_argument(
        '--total', type=int, default=1, help='how many parts exist')
    parser.add_argument(
        '--frame_file', type=str, default='', help='model mode')
        
    args = parser.parse_args()
    return args


def get_selected_frame_indices_by_confidence(data_list, feat_prefix, topk=8):
    frame_indices = {}

    data = open(data_list).readlines()
    data = [x.strip() for x in data]

    for item in data:
        video_id, _, _ = item.split()
        feat_file = video_id + '_loss.pkl'
        filename = osp.join(feat_prefix, feat_file)
        with open(filename, 'rb') as f:
            probs = pickle.load(f)

        topk_idxs = probs.argsort()[-topk:]
        # topk_idxs = topk_idxs[-3:]
        # topk_idxs = np.concatenate([topk_idxs[-2:], topk_idxs[-1:]])
        topk_idxs = np.sort(topk_idxs)[::-1]
        np.random.shuffle(topk_idxs)
        # topk_idxs = probs

        frame_indices[video_id] = topk_idxs
        # frame_indices[video_id] = np.arange(len(probs))

    return frame_indices


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
    args.batch_size = 200

    # frame_indices = get_selected_frame_indices_by_confidence(args.data_list, args.feat_prefix)
    frame_indices = pickle.load(open(args.frame_file, 'rb'))

    # define the data pipeline for Untrimmed Videos
    data_pipeline = [
        dict(type='IndexSampleFrames',
             frame_file=args.frame_file),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 256)),
        dict(type='CenterCrop', crop_size=256),
        dict(type='Normalize', **args.img_norm_cfg),
        dict(type='FormatShape', input_format=args.input_format),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    # define TSN R50 model, the model is used as the feature extractor
    model_cfg = dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNet',
            depth=101,
            in_channels=args.in_channels,
            norm_eval=False),
        cls_head=dict(
            type='TSNHead',
            num_classes=200,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1)),
        test_cfg=dict(average_clips=None, feature_extraction=False))
    model = build_model(model_cfg)
    # load pretrained weight into the feature extractor
    state_dict = torch.load(args.ckpt)['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()

    data = open(args.data_list).readlines()
    data = [x.strip() for x in data]
    data = data[args.part::args.total]

    # finished = [f[:-4] for f in os.listdir(args.output_prefix)]
    # data = [d for d in data if d.split()[0] not in finished]
    # data = data[1300:]

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    # if not osp.exists(args.feat_prefix):
    #     os.system(f'mkdir -p {args.output_prefix}')

    num_correct = 0
    for i, item in enumerate(data):
        frame_dir, length, target = item.split()
        target = torch.Tensor([int(target)]).long().cuda()
        print(frame_dir)
        frame_dir = osp.join(args.data_prefix, frame_dir)
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
        # imgs = imgs.reshape((shape[0], 1) + shape[1:])
        imgs = imgs.unsqueeze(0).cuda()

        def forward_data(model, data, target):
            # chop large data into pieces and extract feature from them
            start_idx = 0
            num_clip = imgs.shape[0]
            # while start_idx < num_clip:
            with torch.no_grad():
                part = imgs#[start_idx:start_idx + args.batch_size]
                feat = model.forward(part, label=target, return_loss=True)
                correct, loss = feat['top1_acc'], feat['loss_cls']
                # start_idx += args.batch_size
            return loss, correct
            # return np.concatenate(results)

        loss, correct = forward_data(model, imgs, target)
        if correct.item() == 1:
            num_correct += 1
        print(f"{num_correct} out of {i+1} are correct, acc: {num_correct / (i+1)}")
        # print(f"{frame_dir} loss: {loss.item()}")
        prog_bar.update()


if __name__ == '__main__':
    main()
