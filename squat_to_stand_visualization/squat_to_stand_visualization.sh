#!/bin/bash
avconv -i squat_to_stand.mov -crf 10 squat_to_stand.mp4
avconv -i squat_to_stand.mp4 \
  -ss 2.83 \
  squat_to_stand_trimmed.mp4
avconv -i squat_to_stand_trimmed.mp4 \
  -r 0.5 \
  -vf crop=644:1846:214:0 \
  -f image2 squat_to_stand_%02d.png

 avconv -i squat_to_stand.mp4 -r 0.5 -f image2 squat_to_stand_%02d.png

mogrify -crop 600x1800+3150+300 -path cropped *.png