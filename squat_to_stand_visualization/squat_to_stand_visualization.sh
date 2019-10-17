#!/bin/bash
 avconv -i crouch_to_stand.webm -crf 10 crouch_to_stand.mp4
avconv -i crouch_to_stand.mp4 \
  -ss 2.83 \
  crouch_to_stand_trimmed.mp4
avconv -i crouch_to_stand_trimmed.mp4 \
  -r 0.5 \
  -vf crop=644:1846:214:0 \
  -f image2 crouch_to_stand_%02d.png



