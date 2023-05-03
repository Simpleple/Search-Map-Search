#!/usr/bin/env bash
cd ../
# python build_rawframes.py ../../data/ActivityNet/videos/ ../../data/ActivityNet/rawframes/ --level 1 --flow-type tvl1 --ext mp4 --task both  --new-short 256
python build_rawframes.py ../../data/ActivityNet/videos/ ../../data/ActivityNet/rawframes/ --level 1 --ext mp4 --task rgb --new-short 256 --use-opencv --resume
python build_rawframes.py ../../data/ActivityNet/videos/ ../../data/ActivityNet/rawframes/ --level 1 --ext mkv --task rgb --new-short 256 --use-opencv --resume
echo "Raw frames (RGB and tv-l1) Generated for train set"

cd activitynet/
