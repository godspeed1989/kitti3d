GT_DIR=/mine/KITTI_DAT/training/label_2

PRED_DIR=/mine/KITTI/kitti3d/backup/0experiments_f3_c6

OUTPUT=$PRED_DIR/log

# start test
`pwd`/evaluate_object_3d_offline $GT_DIR $PRED_DIR > $OUTPUT 2>&1
