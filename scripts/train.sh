GPUS=3
export CUDA_VISIBLE_DEVICES=$GPUS

BASE_ROOT=/Users/nattari/Bielefeld_Work
INPUT_FILES=$BASE_ROOT/bitbucket/project_na/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching
IMAGE_DIR=$BASE_ROOT/Data/CUB_200_2011/CUB_200_2011/images
ANNO_DIR=$BASE_ROOT/bitbucket/project_na/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/data
CKPT_DIR=$BASE_ROOT/bitbucket/project_na/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/data/model_data
LOG_DIR=$BASE_ROOT/data/logs
PRETRAINED_PATH=$BASE_ROOT/bitbucket/project_na/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/pretrained_models/mobilenet.tar
#PRETRAINED_PATH=$BASE_ROOT/resnet50.pth
IMAGE_MODEL=mobilenet_v1
lr=0.0002
num_epoches=300
batch_size=16
lr_decay_ratio=0.9
epoches_decay=80_150_200
num_classes=200

python $INPUT_FILES/train.py \
    --CMPC \
    --CMPM \
    --bidirectional \
    --num_classes $num_classes \
    --model_path $PRETRAINED_PATH \
    --image_model $IMAGE_MODEL \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --checkpoint_dir $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epoches $num_epoches \
    --lr $lr \
    --lr_decay_ratio $lr_decay_ratio \
    --epoches_decay ${epoches_decay}
