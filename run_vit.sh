#!/bin/bash
export HIP_VISIBLE_DEVICES=1
export IREE_BACKEND=rocm
exec .lake/build/bin/vit-tiny-train 2>&1 | tee vit_train.log
