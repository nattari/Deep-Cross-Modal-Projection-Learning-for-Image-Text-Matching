BASE_ROOT=/Users/nattari/Bielefeld_Work

IMAGE_ROOT=$BASE_ROOT/Data/CUB_200_2011/CUB_200_2011/images
JSON_ROOT=$BASE_ROOT/bitbucket/project_na/category_description_no_hypercategory/Data/train_instance_caption.json
OUT_ROOT=$BASE_ROOT/bitbucket/project_na/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/data

echo "Process CUB-Birds dataset and save it as pickle form"

python ${BASE_ROOT}/bitbucket/project_na/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/datasets/preprocess.py \
        --img_root=${IMAGE_ROOT} \
        --json_root=${JSON_ROOT} \
        --out_root=${OUT_ROOT} \
        --min_word_count 3\
        --first 
