# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import copy
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model, Guided


def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    parser.add_argument('--data-prefix', default='data/ActivityNet/rawframes', help='dataset prefix')
    parser.add_argument('--feat-prefix', default='data/ActivityNet/rgb_feat', help='feature prefix')
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
        '--dataset', choices=['ActivityNet', 'FCVID', 'UCF101'], default='ActivityNet', help='video dataset')
    args = parser.parse_args()
    return args


def load_features_and_init_idxs(feat_prefix, video_id, topk=8):
    filename = osp.join(feat_prefix, video_id + '_r50.pkl')
    with open(filename, 'rb') as f:
        features = pickle.load(f)
        frame_probs = features['gt_prob']

    topk_idxs = frame_probs.argmax().repeat(topk)
    topk_idxs = np.sort(topk_idxs)

    return features, topk_idxs


def compute_solution(problem, solution):
    from torch.nn import CrossEntropyLoss
    criterion = CrossEntropyLoss()

    features = problem['features']
    target   = problem['target']
    model    = problem['model']

    features = features[solution]
    features = torch.FloatTensor(features)#.cuda()

    with torch.no_grad():
        logit = model.cls_head.fc_cls(features.mean(dim=0))
        loss = criterion(logit.unsqueeze(0), target).item()

    return loss 


def hierarchical_search(model, target, features, init_frames, num_clips=10, out_frames=8, best=5, mutate_num=3, round=10):
    num_evals = 0
    video_len = features['features'].shape[0]
    clip_len = 30
    num_clips = video_len // clip_len

    # solve for clip combinations
    clip_frame_idxs = [np.arange(i * clip_len, (i + 1) * clip_len) for i in np.arange(num_clips)]
    clip_frame_idxs[-1] = np.arange((num_clips - 1) * clip_len, video_len)
    clip_features = np.array([np.mean(features['features'][idxs], axis=0) for idxs in clip_frame_idxs])

    init_clip_idxs = np.array([min(f // clip_len, num_clips - 1) for f in init_frames])  # np.random.choice(num_clips, out_frames, replace=True)

    problem = {'model': model, 'n_frames': out_frames, 'features': clip_features, 'target': target, 'stage': 1}
    params = dict(solution=init_clip_idxs,
                method='first-delta-improvement',
                n_iter=1,
                n_epoch=3,
                mu=1,
                patience=1,
                verbose=False)

    alg = Guided(problem)
    alg.set_params(params)
    alg.refresh_params()
    alg_solution = alg.solve()
    history = alg.get_history()

    num_evals += alg.num_evals

    # solve for frame combinations
    problem = {'model': model, 'n_frames': out_frames, 'features': features['features'], 'target': target, 'stage': 1}
    clip_idxs = alg_solution
    frame_idxs = np.array([np.random.choice(clip_frame_idxs[i]) for i in clip_idxs])
    best_frame_idx, best_perf = frame_idxs, np.Infinity
    for j in range(out_frames):
        mutated_frames = clip_frame_idxs[clip_idxs[j]]
        for f in mutated_frames:
            if f == frame_idxs[j]:
                continue
            new_frame_idxs = copy.deepcopy(best_frame_idx)
            new_frame_idxs[j] = f
            perf = compute_solution(problem, new_frame_idxs)
            num_evals += 1
            if perf < best_perf:
                best_frame_idx = new_frame_idxs
                best_perf = perf

    return best_frame_idx, best_perf, num_evals


def main(args):

    if args.dataset == 'ActivityNet':
        num_classes = 200
    elif args.dataset == 'FCVID':
        num_classes = 239
    elif args.dataset == 'UCF101':
        num_classes = 101

    # define TSN R50 model, the model is used as the feature extractor
    model_cfg = dict(
        type='Recognizer2D',
        backbone=dict(
            type='ResNet',
            depth=50,
            in_channels=args.in_channels,
            norm_eval=False),
        cls_head=dict(
            type='TSNHead',
            num_classes=num_classes,
            in_channels=2048,
            spatial_type='avg',
            consensus=dict(type='AvgConsensus', dim=1)),
        test_cfg=dict(average_clips=None, feature_extraction=False))
    model = build_model(model_cfg)
    # load pretrained weight into the feature extractor
    state_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))['state_dict']
    model.load_state_dict(state_dict)
    model = model#.cuda()
    model.eval()

    data = open(args.data_list).readlines()
    data = [x.strip() for x in data]
    if args.end < 0:
        data = data[args.start:]
    else:
        data = data[args.start:args.end]
    print(f'{len(data)} videos to search')

    frame_results = {}

    for item in data:
        video_id, length, target = item.split()
        video_id = osp.basename(video_id)
        length, target = int(length), int(target)
        target = torch.Tensor([target]).long()#.cuda()
        frame_dir = osp.join(args.data_prefix, video_id)

        search_res = {}
        features, best_frame_seq = load_features_and_init_idxs(args.feat_prefix, video_id)
        problem = {'model': model, 'n_frames': 8, 'features': features['features'], 'target': target, 'stage': 1}

        frame_result, score, num_evaluations = hierarchical_search(model, target, features, best_frame_seq)
        print(f"\t{video_id}\thier score: {score}\tnum_evaluations: {num_evaluations}")

    with open(args.feat_prefix + f"greedy_res_{args.start}_{args.end}_gls_r50.pkl", 'wb') as f:
        pickle.dump(frame_results, f)

    return 


if __name__ == '__main__':
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

    main(args)
