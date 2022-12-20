python doccano.py --doccano_file ./data/doccano_ext.json --task_type ext --save_dir ./data --splits 0.8 0.2 0 --schema_lang ch

```bash
python -u -m paddle.distributed.launch --cpu finetune.py ^
    --device cpu ^
    --logging_steps 10 ^
    --save_steps 100 ^
    --eval_steps 100 ^
    --seed 42 ^
    --model_name_or_path uie-base ^
    --output_dir $finetuned_model ^
    --train_path data/train.txt ^
    --dev_path data/dev.txt  ^
    --max_seq_length 512  ^
    --per_device_eval_batch_size 16 ^
    --per_device_train_batch_size  16 ^
    --num_train_epochs 100 ^
    --learning_rate 1e-5 ^
    --do_train ^
    --do_eval ^
    --do_export ^
    --export_model_dir ./checkpoint/model_best ^
    --label_names 'start_positions' 'end_positions' ^
    --overwrite_output_dir ^
    --disable_tqdm True ^
    --metric_for_best_model eval_f1 ^
    --load_best_model_at_end  True ^
    --save_total_limit 1 
```
