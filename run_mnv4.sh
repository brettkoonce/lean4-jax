#!/bin/bash
export HIP_VISIBLE_DEVICES=1
export IREE_BACKEND=rocm
exec .lake/build/bin/mobilenet-v4-train 2>&1 | tee mnv4_train.log
