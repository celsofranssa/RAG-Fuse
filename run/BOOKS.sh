# overrides
data=BOOKS
model=RetrieverBERT

text_max_length=256
label_max_length=256
label_enhancement=LLM
text_features_source=TXT

## sparse_retrieve
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[sparse_retrieve] \
    model=BM25 \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/sparse_retrieve_${data}_${fold_idx}.tmr
done

# prompt_opt (uncomment to run — requires vLLM server on localhost:8001)
#time_start=$(date '+%Y-%m-%d %H:%M:%S')
#python main.py \
#  tasks=[prompt_opt] \
#  data=$data \
#  data.text_features_source=$text_features_source
#time_end=$(date '+%Y-%m-%d %H:%M:%S')
#echo "$time_start,$time_end" > resource/time/prompt_opt_${data}.tmr

# label_desc (uncomment to run — requires vLLM server + optimized_prompt.txt)
#for fold_idx in $(seq $1 $2);
#do
#  time_start=$(date '+%Y-%m-%d %H:%M:%S')
#  python main.py \
#    tasks=[label_desc] \
#    data=$data \
#    data.text_features_source=$text_features_source \
#    data.folds=[$fold_idx]
#  time_end=$(date '+%Y-%m-%d %H:%M:%S')
#  echo "$time_start,$time_end" > resource/time/label_desc_${data}_${fold_idx}.tmr
#done

# dense_retrieve fit
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fit] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=64 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fit_LLM_${model}_${data}_${fold_idx}.tmr
done

# dense_retrieve predict
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[predict] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=128 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/predict_LLM_${model}_${data}_${fold_idx}.tmr
done

# dense_retrieve eval
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[eval] \
    trainer.max_epochs=5 \
    trainer.patience=3 \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=128 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/eval_LLM_${model}_${data}_${fold_idx}.tmr
done

# fuse
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[fuse] \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_features_source=$text_features_source \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/fuse_LLM_${model}_${data}_${fold_idx}.tmr
done


# aggregate
for fold_idx in $(seq $1 $2);
do
  time_start=$(date '+%Y-%m-%d %H:%M:%S')
  python main.py \
    tasks=[aggregate] \
    model=$model \
    model.name=LLM_${model} \
    data=$data \
    data.text_max_length=$text_max_length \
    data.label_max_length=$label_max_length \
    data.label_enhancement=$label_enhancement \
    data.text_features_source=$text_features_source \
    data.batch_size=128 \
    data.num_workers=12 \
    data.folds=[$fold_idx]
  time_end=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$time_start,$time_end" > resource/time/aggregate_LLM_${model}_${data}_${fold_idx}.tmr
done
