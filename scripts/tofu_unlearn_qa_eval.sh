#!/bin/bash


export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

models=(
    "Llama-2-7b-chat-hf"
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.1-8B-Instruct"
)
forget_retain_splits=(
    # "forget01 retain99"
    # "forget05 retain95"
    "forget10 retain90"
)

per_device_train_batch_size=4 # on two gpus would make effective batch size 32
gradient_accumulation_steps=4

finetune_base_path=data_2_old/relu/llm_weights/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01_datasetqa_forget10/checkpoint-1250
unlearn_base_path=${finetune_base_path}/qa_entropy_all_1e-05_full_5_forget10_layer-15-current_linear
checkpoints=(
    ${unlearn_base_path}/checkpoint-25
    ${unlearn_base_path}/checkpoint-50
    ${unlearn_base_path}/checkpoint-75
    ${unlearn_base_path}/checkpoint-100
    ${unlearn_base_path}/checkpoint-125
    # ${unlearn_base_path}/checkpoint-150
    # ${unlearn_base_path}/checkpoint-175
    # ${unlearn_base_path}/checkpoint-200
    # ${unlearn_base_path}/checkpoint-225
    # ${unlearn_base_path}/checkpoint-250
    # ${unlearn_base_path}/checkpoint-275
    # ${unlearn_base_path}/checkpoint-300
    # ${unlearn_base_path}/checkpoint-325
    # ${unlearn_base_path}/checkpoint-350
    # ${unlearn_base_path}/checkpoint-375
    # ${unlearn_base_path}/checkpoint-400
    # ${unlearn_base_path}/checkpoint-425
    # ${unlearn_base_path}/checkpoint-450
    # ${unlearn_base_path}/checkpoint-475
    # ${unlearn_base_path}/checkpoint-500
    # ${unlearn_base_path}/checkpoint-525
    # ${unlearn_base_path}/checkpoint-550
    # ${unlearn_base_path}/checkpoint-575
    # ${unlearn_base_path}/checkpoint-600
    # ${unlearn_base_path}/checkpoint-625
    # ${unlearn_base_path}/checkpoint-650
    # ${unlearn_base_path}/checkpoint-675
    # ${unlearn_base_path}/checkpoint-700
    # ${unlearn_base_path}/checkpoint-725
    # ${unlearn_base_path}/checkpoint-750
    # ${unlearn_base_path}/checkpoint-775
    # ${unlearn_base_path}/checkpoint-800
    # ${unlearn_base_path}/checkpoint-825
    # ${unlearn_base_path}/checkpoint-850
    # ${unlearn_base_path}/checkpoint-875
    # ${unlearn_base_path}/checkpoint-900
    # ${unlearn_base_path}/checkpoint-925
    # ${unlearn_base_path}/checkpoint-950
    # ${unlearn_base_path}/checkpoint-975
    # ${unlearn_base_path}/checkpoint-1000
    # ${unlearn_base_path}/checkpoint-1025
    # ${unlearn_base_path}/checkpoint-1050
    # ${unlearn_base_path}/checkpoint-1075
    # ${unlearn_base_path}/checkpoint-1100
    # ${unlearn_base_path}/checkpoint-1125
    # ${unlearn_base_path}/checkpoint-1150
    # ${unlearn_base_path}/checkpoint-1175
    # ${unlearn_base_path}/checkpoint-1200
    # ${unlearn_base_path}/checkpoint-1225
    # ${unlearn_base_path}/checkpoint-1250
)


########################################################################################################################
########################################### Unlearn TOFU models ########################################################
########################################################################################################################


for split in "${forget_retain_splits[@]}"; do
    forget_split=$(echo $split | cut -d' ' -f1)
    retain_split=$(echo $split | cut -d' ' -f2)
    for model in "${models[@]}"; do
        # for trainer_experiment in "${trainers_experiments[@]}"; do
            # trainer=$(echo $trainer_experiment | cut -d' ' -f1)
            # experiment=$(echo $trainer_experiment | cut -d' ' -f2)

            for checkpoint in "${checkpoints[@]}"; do
            
                task_name=tofu_${model}_${forget_split}_qa_eval_4
                # model_path=data_2/relu/llm_weights/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01_datasetqa_forget10/checkpoint-1250
                # model_path=data_2/relu/llm_weights/ft_epoch5_lr1e-05_llama2-7b_retain90_wd0.01_datasetqa_forget10/checkpoint-1124
                model_path=${checkpoint}
                echo ${task_name}: Evaluating ${model_path}

                # Eval
                CUDA_VISIBLE_DEVICES=0 python src/eval.py \
                experiment=eval/tofu/default.yaml \
                forget_split=${forget_split} \
                model=${model} \
                task_name=${task_name} \
                model.model_args.pretrained_model_name_or_path=${model_path} \
                paths.output_dir=saves/unlearn/${forget_split}/${task_name}/${checkpoint##*/}/evals \
                retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
                # eval.tofu.overwrite=true

        done
    done
done
