# Copyright (c) OpenMMLab. All rights reserved.
# --------------------------------------

import argparse
import os
import os.path as osp
import pickle
import time

import mmcv
import numpy as np
import torch
import random

import copy
from scipy import spatial

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model, MappingFunctionTrans


class FeatureMappingDataset:

    def __init__(self, data_root, video_ids, portion=1):
        self.data_root = data_root
        self.video_ids = video_ids
        self.portion = portion

    def __getitem__(self, i):
        video_id = self.video_ids[i]

        feat_file = osp.join(self.data_root, video_id + '_r50.pkl')
        with open(feat_file, 'rb') as f:
            features = pickle.load(f)['features']
        
        if self.portion <= 1:
            num_frames = int(features.shape[0] * self.portion)
        else:
            num_frames = int(self.portion)

        total = features.shape[0]
        if total >= num_frames:
            frame_idx = int(total / num_frames // 2)
            frame_idx += total / num_frames * np.arange(num_frames)
            frame_idx = np.array([int(t) for t in frame_idx])
        else:
            frame_idx = np.sort(np.concatenate((np.arange(total), np.floor(
                total/(num_frames-total) * np.arange(num_frames-total)).astype(np.int))))

        features = features[frame_idx]

        return features, video_id, frame_idx

    def __len__(self):
        return len(self.video_ids)


def collate_fn(data):
    features, video_ids, frame_idx = zip(*data)
    batch_size = len(features)
    feat_dim = features[0].shape[1]
    seq_lens = [feat.shape[0] for feat in features]
    max_seq_len = max(seq_lens)
    padded_features = []
    for i in range(batch_size):
        padded_feat = np.concatenate(
            [features[i],
             np.zeros((max_seq_len - seq_lens[i], feat_dim))],
            axis=0)
        padded_features.append(padded_feat)

    feature_tensor = torch.FloatTensor(padded_features)

    mask = torch.arange(max_seq_len).repeat(batch_size, 1)
    mask = mask >= torch.Tensor(seq_lens).view(-1, 1)

    return feature_tensor, mask, video_ids, frame_idx


def search_for_combinations(features, target_feature, frame_idx, out_frames=8, sim_fn=torch.nn.functional.cosine_similarity, round=1):
    ''' greedy search '''
    num_frames = features.shape[0]
    feature_dim = features.shape[1]

    # zero init
    best_comb = []
    best_sim = -np.Infinity

    target_feature = target_feature.unsqueeze(0)

    for r in range(round):
        for i in range(out_frames):
            if r == 0:
                if len(best_comb) > 0:
                    candidate = (features[best_comb].sum(dim=0, keepdim=True) + features) / (len(best_comb) + 1)
                else:
                    candidate = features
                similarities = sim_fn(target_feature, candidate)
                chosen = similarities.argmax().item()
                sim = similarities[chosen].item()
                # if sim > best_sim:
                #     best_sim = sim
                best_comb.append(chosen)
            else:
                popped = best_comb.pop(0)
                candidate = (features[best_comb].sum(dim=0, keepdim=True) + features) / (len(best_comb) + 1)
                similarities = sim_fn(target_feature, candidate)
                chosen = similarities.argmax().item()
                sim = similarities[chosen].item()
                if sim > best_sim:
                    best_sim = sim
                    best_comb.append(chosen)
                else:
                    best_comb.append(popped)

    return frame_idx[np.array(best_comb)]


def parse_args():
    parser = argparse.ArgumentParser(description='Infer frames')
    parser.add_argument('--data-prefix', default='../mmaction2/data/ActivityNet/rawframes', help='dataset prefix')
    # parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument('--data-list', default='../mmaction2/data/ActivityNet/anet_val_video.txt',
                        help='video list of the dataset, the format should be '
                        '`frame_dir num_frames output_file`')
    parser.add_argument('--num-frames',
                        type=int,
                        default=8,
                        help='number of frames sampled')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--input-dim',
                        type=int,
                        default=2048,
                        help='frame input feature dimension')
    parser.add_argument('--feature-prefix',
                        type=str,
                        default='../mmaction2/data/ActivityNet/rgb_feat',
                        help='feature prefix')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size')
    parser.add_argument(
        '--model_path', type=str, default='checkpoints/mapping_bs_128_lr_0.0001/best_pred_ckpt.pt', help='resume training on checkpoint')
    parser.add_argument(
        '--round', type=int, default=1)
    parser.add_argument(
        '--distance', choices=['cosine', 'l2'], default='cosine')
    parser.add_argument(
        '--frame_portion', type=float, default=1, help='portion of raw frames that are used for inference. percentage if value <= 1, else as number of frames')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    args.is_rgb = args.modality == 'RGB'
    args.clip_len = 1 if args.is_rgb else 5
    args.input_format = 'NCHW' if args.is_rgb else 'NCHW_Flow'
    rgb_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)

    frame_selector = MappingFunctionTrans(feature_dim=args.input_dim).cuda()

    state_dict = torch.load(args.model_path)
    frame_selector.load_state_dict(state_dict)

    video_ids = open(args.data_list).readlines()
    video_ids = [x.strip().split()[0] for x in video_ids]
    
    dataset = FeatureMappingDataset(data_root=args.feature_prefix,
                                    video_ids=video_ids,
                                    portion=args.frame_portion)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True,
                                             collate_fn=collate_fn,
                                             num_workers=16)
    frame_res = {}

    prog_bar = mmcv.ProgressBar(len(dataset))
    
    frame_selector.eval()
    with torch.no_grad():
        for step, (features, masks, video_ids, frame_idxs) in enumerate(dataloader):
            features, masks = features.cuda(), masks.cuda()
            out = frame_selector(features, masks)
            for i in range(len(video_ids)):
                frame = search_for_combinations(features[i][:frame_idxs[i].shape[0]], out[i], frame_idxs[i], round=args.round, out_frames=args.num_frames)
                frame_res[video_ids[i]] = frame
                prog_bar.update()

    with open(args.model_path[:-4] + f'_pred_{args.frame_portion}_out_{args.num_frames}.pkl', 'wb') as f:
        pickle.dump(frame_res, f)


if __name__ == "__main__":
    main()