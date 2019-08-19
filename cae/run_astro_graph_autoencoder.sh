#!/bin/bash
EXE="graph_autoencoder_astro.py"

# may or may not need this - seems sketchy...
export KMP_DUPLICATE_LIB_OK=True

TRAIN_STEPS=50
BATCH_SIZE=100
NUM_EPOCHS=5
DATA_DIR="/Users/perdue/Dropbox/Quantum_Computing/hep-qml/data/cae_splits"
DATA_TYPE="stargalaxy_sim_20190214"
MODEL_DIR="/tmp/${DATA_TYPE}"
LOG_LEVEL="DEBUG"

LEARNING_RATE="0.00005"
LEARNING_RATE="0.00001"
LEARNING_RATE="0.0001"

ARGS="--data-dir ${DATA_DIR}"
ARGS+=" --data-type ${DATA_TYPE}"
ARGS+=" --model-dir ${MODEL_DIR}"
ARGS+=" --learning-rate ${LEARNING_RATE}"
ARGS+=" --num-epochs ${NUM_EPOCHS}"
ARGS+=" --log-level ${LOG_LEVEL}"
# ARGS+=" --train-steps ${TRAIN_STEPS}"

cat << EOF
python $EXE $ARGS
EOF

python $EXE $ARGS

echo -e "\a"
