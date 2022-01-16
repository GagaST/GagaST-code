# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

TAG=$1        # num of GPUs to use
#CODE_ROOT=$3   # path/to/code_root
OUTPUT_DIR=$2  # output dir to save checkpoints, decodings, etc
PARA_ROOT=$3   # path/to/XGLUE/NTG
MONO_ROOT=$4
FT_BIN=$5

PARA_BIN=$PARA_ROOT/bin
PARA_REF=$PARA_ROOT/ref

langs=en,zh
pairs=en-zh,zh-en
ft_langs=en-zh,zh-en
mono_domain=LYRICS
para_domain=WMT
ft_domain=LYRICS

lr=$6

TBS=1024
max_tokens=$7
update_freq=$8

mono_ratio=$9
warmup=4000
mepoch=${10}
prefix=${11}

#word_shuffle=3
#word_dropout=0.1
#word_blank=0.1
word_shuffle=0
word_dropout=0.0
word_blank=0.0

mask_rate=0.3
poisson_lbd=3.5

task=xdae_multilingual_translation
EXP="Pretrain_all_musescore_filtered_single_tag_lr${lr}_m${mepoch}_r${mono_ratio}_mtoken${max_tokens}_upf${update_freq}_M${TAG}"

SAVE=${OUTPUT_DIR}/$EXP
LOG=$SAVE/log

mkdir -p $SAVE
mkdir -p $LOG

SUFFIX=""
#if [ ! -f $SAVE/checkpoint_last.pt ]; then
   #echo "copy pretrained model to last"
   #cp $PRETRAIN $SAVE/checkpoint_last.pt
#fi

if [ ! -f $SAVE/checkpoint1.pt ]; then
   SUFFIX="$SUFFIX --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer"
fi

NOW=`date '+%F_%H_%M_%S'`
fairseq-train ${MONO_ROOT}:${PARA_BIN}  \
           --finetune-data $FT_BIN \
           --save-dir $SAVE \
           --arch mbart_base \
           --encoder-layers 12 \
           --decoder-layers 12 \
           --langs $langs \
           --max-source-positions 512 \
           --max-target-positions 512 \
           --layernorm-embedding \
           --skip-invalid-size-inputs-valid-test \
           --task $task \
           --with-len \
           --mono-langs $langs \
           --para-langs $pairs \
           --finetune-langs $ft_langs \
           --domains LYRICS,WMT \
           --mono-domain $mono_domain \
           --para-domain $para_domain \
           --ft-domain $ft_domain \
           --add-lang-token \
           --sample-break-mode eos \
           --mono-ratio $mono_ratio \
           --downsample-by-min \
           --mask $mask_rate --replace-length 1 \
           --mask-length span-poisson \
           --poisson-lambda $poisson_lbd \
           --rotate 0 \
           --word-shuffle $word_shuffle \
           --word-dropout $word_dropout \
           --word-blank $word_blank \
           --criterion label_smoothed_cross_entropy \
           --ignore-prefix-size $prefix \
           --label-smoothing 0.2  \
           --placeholder 0 \
           --dataset-impl mmap \
           --optimizer adam \
           --adam-eps 1e-06 \
           --adam-betas '(0.9, 0.98)' \
           --lr-scheduler inverse_sqrt \
           --lr $lr --stop-min-lr 1e-09 \
           --warmup-init-lr 1e-07 \
           --warmup-updates $warmup \
           --dropout 0.1 \
           --attention-dropout 0.1  \
           --weight-decay 0.01 \
           --max-tokens $max_tokens \
           --update-freq $update_freq \
           --save-interval 1 \
           --save-interval-updates 100000 \
           --keep-interval-updates 1 \
           --max-epoch $mepoch \
           --seed 1023 \
           --log-format simple --log-interval 5 \
           --ddp-backend no_c10d --fp16 \
           --tensorboard-logdir $SAVE \
           $SUFFIX 2>&1 | tee $LOG/log_$NOW.txt


#python $CODE_ROOT/evaluation/decode_all_bis.py --task $task --beam 3 --start ${decode_from} --ngpu ${NGPU} --epoch ${mepoch} \
           #--split valid --exp $EXP --data_path $DATA_BIN --spe $SPE --save_dir $OUTPUT_DIR --dataset QG \
           #--code_root $CODE_ROOT

#python $CODE_ROOT/evaluation/eval_exp.py --task $task --test_beam 10 --ngpu ${NGPU} --epoch ${mepoch} \
           #--exp $EXP --data_path $DATA_BIN --ref_folder $DATA_REF \
           #--lgs en-fr-es-de-it-pt --supervised_lg $lg --spe $SPE --save_dir $OUTPUT_DIR --dataset QG \
#           --code_root $CODE_ROOT
