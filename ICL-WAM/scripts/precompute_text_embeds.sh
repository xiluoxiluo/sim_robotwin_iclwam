cd /data/share/1919650160032350208/zjj/ICL-WAM
export DIFFSYNTH_MODEL_BASE_PATH="/data/share/1919650160032350208/zjj/fastwam/checkpoints"
export PYTHONPATH="/data/share/1919650160032350208/zjj/ICL-WAM/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/precompute_text_embeds.py \
  --config-name train \
  task=robotwin_zeva_fastwam_3cam_384 \
  +overwrite=false \
  data.train.text_embedding_cache_dir=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/text_embeds_cache_new \
  data.val.text_embedding_cache_dir=/data/share/1919650160032350208/foundation_model/datasets/robotwin2.0/text_embeds_cache_new