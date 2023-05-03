# Copyright (c) OpenMMLab. All rights reserved.
# --------------------------------------

import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch
import random
import torch.nn.functional as F

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model, MappingFunctionTrans


class FeatureMappingDataset:

    def __init__(self, data_root, video_ids, label_file, num_frames=16):
        self.data_root = data_root
        self.video_ids = video_ids
        self.load_labels(label_file)
        self.num_frames = num_frames

    def load_labels(self, file):
        with open(file, 'rb') as f:
            self.label_dict = pickle.load(f)

    def __getitem__(self, i):
        video_id = self.video_ids[i]

        feat_file = osp.join(self.data_root, video_id + '_r50.pkl')
        with open(feat_file, 'rb') as f:
            features = pickle.load(f)['features']
        indices = self.label_dict[video_id]
        target_feature = features[indices].mean(axis=0)

        total = features.shape[0]
        if total >= self.num_frames:
            frame_idx = int(total / self.num_frames // 2)
            frame_idx += total / self.num_frames * np.arange(self.num_frames)
            frame_idx = np.array([int(t) for t in frame_idx])
        else:
            frame_idx = np.arange(total)

        features = features[frame_idx]

        return features, target_feature

    def __len__(self):
        return len(self.video_ids)


def collate_fn(data):
    features, target_features = zip(*data)
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

    target_tensor = torch.FloatTensor(np.stack(target_features))

    return feature_tensor, mask, target_tensor


def parse_args():
    parser = argparse.ArgumentParser(description='Train mapping function')
    parser.add_argument('--data-prefix', default='../mmaction2/data/ActivityNet/rawframes', help='dataset prefix')
    parser.add_argument('--data-list', default='../mmaction2/data/ActivityNet/anet_train_video.txt',
                        help='video list of the dataset, the format should be '
                        '`frame_dir num_frames output_file`')
    parser.add_argument('--num-frames',
                        type=int,
                        default=16,
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
    parser.add_argument('--num-epochs',
                        type=int,
                        default=100,
                        help='feature prefix')
    parser.add_argument(
        '--label_file',
        type=str,
        default='../mmaction2/data/ActivityNet/rgb_featgreedy_res_0_-1_gls_r50.pkl')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument(
        '--distance', choices=['cosine', 'l2'], default='cosine')
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

    if args.distance == 'l2':
        criterion = torch.nn.MSELoss()
    elif args.distance == 'cosine':
        criterion = lambda feat1, feat2 : -F.cosine_similarity(feat1, feat2).mean()
    selector_optimizer = torch.optim.Adam(frame_selector.parameters(),
                                          lr=args.lr,
                                          betas=(0.5, 0.999))

    video_ids = open(args.data_list).readlines()
    video_ids = [x.strip().split()[0] for x in video_ids]
    # random.shuffle(video_ids)
    num_samples = len(video_ids)
    train_ids = video_ids[:int(0.8 * num_samples)]
    val_ids = video_ids[int(0.8 * num_samples):]

    train_dataset = FeatureMappingDataset(data_root=args.feature_prefix,
                                          video_ids=train_ids,
                                          label_file=args.label_file,
                                          num_frames=args.num_frames)
    val_dataset = FeatureMappingDataset(data_root=args.feature_prefix,
                                        video_ids=val_ids,
                                        label_file=args.label_file,
                                        num_frames=args.num_frames)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn,
                                               num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             collate_fn=collate_fn,
                                             num_workers=16)

    best_val_loss = np.Infinity
    for epoch in range(start_epoch, args.num_epochs):

        train_loss = 0
        frame_selector.eval()
        for step, (input, mask, target) in enumerate(train_loader):
            input, mask, target = input.cuda(), mask.cuda(), target.cuda()
            out = frame_selector(input, mask)
            loss = criterion(out, target)

            selector_optimizer.zero_grad()
            loss.backward()
            selector_optimizer.step()

            train_loss += loss.item() * input.shape[0]

        print('epoch: %d, train loss: %.6f' %
              (epoch, train_loss / len(train_dataset)))

        frame_selector.eval()
        valid_loss = 0
        targets, preds = [], []
        with torch.no_grad():
            for step, (input, mask, target) in enumerate(val_loader):
                input, mask, target = input.cuda(), mask.cuda(), target.cuda()
                out = frame_selector(input, mask)
                loss = criterion(out, target)
                valid_loss += loss.item() * input.shape[0]

        avg_val_loss = valid_loss / len(val_dataset)
        print(f'epoch: {epoch}, valid loss: {avg_val_loss:.6f}\n')

        save_path = f"checkpoints/mapping_trans_{args.num_frames}_bs_{args.batch_size}_lr_{args.lr}_{args.distance}"
        if not osp.exists(save_path):
            os.makedirs(save_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(frame_selector.state_dict(), save_path + f'/best_pred_ckpt.pt')

        torch.save(frame_selector.state_dict(),
                   save_path + f'/epoch_{epoch}_loss_{avg_val_loss:.6f}.pt')


if __name__ == "__main__":

    main()