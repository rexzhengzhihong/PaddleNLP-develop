python doccano.py \
--doccano_file /home/DiskA/zncsPython/paddlenlp/uie/data/doccano_ext.jsonl \
--task_type ext \
--save_dir /home/DiskA/zncsPython/paddlenlp/uie/data \
--splits 0.8 0.2 0 \
--schema_lang ch


```shell
export train_path=/home/DiskA/zncsPython/paddlenlp/uie
export finetuned_model=/home/DiskA/zncsPython/paddlenlp/uie/ 
export path_model=/home/DiskA/zncsPython/paddlenlp/uie/uie-base
python  finetune.py \
    --device cpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 1000 \
    --model_name_or_path uie-base \
    --output_dir $finetuned_model \
    --train_path $train_path/data/train.txt \
    --dev_path $train_path/data/dev.txt  \
    --max_seq_length 512  \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size  16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir $finetuned_model \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1 
```
```shell

命令用不了
export train_path=/home/DiskA/zncsPython/paddlenlp/uie
export finetuned_model=/home/DiskA/zncsPython/paddlenlp/uie/ 
export path_model=/home/DiskA/zncsPython/paddlenlp/uie/uie-base
python finetune.py \
    --train_path $train_path/data/train.txt \
    --dev_path $train_path/data/dev.txt \
    --output_dir $finetuned_model \
    --learning_rate 1e-5 \
    --batch_size 16 \
    --max_seq_len 256 \
    --num_train_epochs 100 \
    --model uie-base \
    --seed 1000 \
    --logging_steps 10 \
    --valid_steps 100 \
    --device gpu 
```


```shell
python evaluate.py \
    --model_path /home/DiskA/zncsPython/paddlenlp/uie/checkpoint/model_best \
    --test_path /home/DiskA/zncsPython/paddlenlp/uie/data/dev.txt \
    --batch_size 16 \
    --max_seq_len 512 
```

python evaluate.py \
    --model_path /home/DiskA/zncsPython/paddlenlp/uie/checkpoint/model_best \
    --test_path /home/DiskA/zncsPython/paddlenlp/uie/data/dev.txt \
    --debug