#! /bin/bash

set -euo pipefail

## unfolds s4 audio_wav dir
for dir in ./s4/audio_wav/train/*/
do
    mv $dir* ./s4/audio_wav/
    rm -r $dir
done
rm -r ./s4/audio_wav/train/

for dir in ./s4/audio_wav/val/*/
do
    mv $dir* ./s4/audio_wav/
    rm -r $dir
done
rm -r ./s4/audio_wav/val/

for dir in ./s4/audio_wav/test/*/
do
    mv $dir* ./s4/audio_wav/
    rm -r $dir
done
rm -r ./s4/audio_wav/test/


## unfolds s4 gt_masks
for dir in ./s4/gt_masks/test/*/
do
    mv $dir* ./s4/gt_masks/
    rm -r $dir
done
rm -r ./s4/gt_masks/test/

for dir in ./s4/gt_masks/train/*/
do
    mv $dir* ./s4/gt_masks/
    rm -r $dir
done
rm -r ./s4/gt_masks/train/

for dir in ./s4/gt_masks/val/*/
do
    mv $dir* ./s4/gt_masks/
    rm -r $dir
done
rm -r ./s4/gt_masks/val/

for dir in ./s4/gt_masks/*/
do
    mv $dir* ./s4/gt_masks/
    rm -r $dir
done


## unfolds s4 visual_frames
for dir in ./s4/visual_frames/test/*/
do
    mv $dir* ./s4/visual_frames/
    rm -r $dir
done
rm -r ./s4/visual_frames/test/

for dir in ./s4/visual_frames/train/*/
do
    mv $dir* ./s4/visual_frames/
    rm -r $dir
done
rm -r ./s4/visual_frames/train/

for dir in ./s4/visual_frames/val/*/
do
    mv $dir* ./s4/visual_frames/
    rm -r $dir
done
rm -r ./s4/visual_frames/val/

for dir in ./s4/visual_frames/*/
do
    mv $dir* ./s4/visual_frames/
    rm -r $dir
done


## unfolds ms3 audio_wav
mv ms3/audio_wav/train/* ms3/audio_wav
mv ms3/audio_wav/test/* ms3/audio_wav
mv ms3/audio_wav/val/* ms3/audio_wav

rm -r ms3/audio_wav/train
rm -r ms3/audio_wav/test
rm -r ms3/audio_wav/val

# ## unfolds ms3 gt_masks
for dir in ./ms3/gt_masks/test/*/
do
    mv $dir* ./ms3/gt_masks/
    rm -r $dir
done
rm -r ./ms3/gt_masks/test/

for dir in ./ms3/gt_masks/train/*/
do
    mv $dir* ./ms3/gt_masks/
    rm -r $dir
done
rm -r ./ms3/gt_masks/train/

for dir in ./ms3/gt_masks/val/*/
do
    mv $dir* ./ms3/gt_masks/
    rm -r $dir
done
rm -r ./ms3/gt_masks/val/

## unfolds ms3 visual_frames
for dir in ./ms3/visual_frames/*/
do
    mv $dir* ./ms3/visual_frames/
    rm -r $dir
done