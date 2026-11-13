#!/bin/sh
#
# Copyright (c) 2025 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.

weston_user=$(ps aux | grep '/usr/bin/weston '|grep -v 'grep'|awk '{print $1}')
FRAMEWORK=$1
DEPLOY_PATH=$2
echo "stai_mpu wrapper used : "$FRAMEWORK
CONFIG=$(find $DEPLOY_PATH/Resources -name "config_board.sh")
source $CONFIG
cmd="python3 $DEPLOY_PATH/Application/object_detection.py -m $DEPLOY_PATH/$OBJ_DETEC_MODEL -l $DEPLOY_PATH/Resources/$OBJ_DETEC_MODEL_LABEL.txt --framerate $DFPS --frame_width $DWIDTH --frame_height $DHEIGHT --camera_src $CAMERA_SRC"

if [ "$weston_user" != "root" ]; then
	echo "user : "$weston_user
	script -qc "su -l $weston_user -c '$cmd'"
else
	$cmd
fi