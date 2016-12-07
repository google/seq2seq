#! /usr/bin/env bash

if [ -z ${MODEL_DIR+x} ]; then
  echo "Please set MODEL_DIR environment variable"
  exit 1
fi

if [ -z ${TFPROF_PATH+x} ]; then
  echo "Please set TFPROF_PATH environment variable"
  exit 1
fi

TFPROF_ARGS=""
TFPROF_ARGS+=" --graph_path=${MODEL_DIR}/graph.pbtxt"
TFPROF_ARGS+=" --run_meta_path=${MODEL_DIR}/metadata/run_meta"
TFPROF_ARGS+=" --checkpoint_path=${MODEL_DIR}/model.ckpt-1-00000-of-00001"
TFPROF_ARGS+=" --op_log_path=${MODEL_DIR}/metadata/tfprof_log"
TFPROF_ARGS+=" --max_depth=10"

TFPROF_CMD="${TFPROF_PATH} ${TFPROF_ARGS}"

$TFPROF_CMD <<< "scope -select params,bytes -order_by params"
echo -e "\n"
$TFPROF_CMD <<< "graph -select micros -min_micros 1000 -max_depth 100 -account_type_regexes .* -order_by micros"
echo -e "\n"
$TFPROF_CMD <<< "scope -select float_ops -min_float_ops 1 -account_type_regexes .* -order_by float_ops"
echo -e "\n"
$TFPROF_CMD <<< "scope -select device -account_type_regexes .* -max_depth 6 -start_name_regexes model"
echo -e "\n"

exit 0