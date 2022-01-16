# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

gid=$1
source_lg=$2
target_lg=$3
DATA=$4
BPE_CODES=$5
ft_domain=$6
prefix_size=$7
SPLIT=$8
model=$9
beam_size=${10}
lyrics_dict=${11}
notes_weight=${12}
shape_weight=${13}
durations_weight=${14}
rests_weight=${15}
min_align_prob=${16}
SAVE_PATH=${17}
placeholder=0

mkdir -p $SAVE_PATH

task=xdae_multilingual_translation

suffix=""
if [[ $# -eq 18 && ${18} == "with_group" ]]; then
    suffix="--with-predefined-notes-group"
fi


CUDA_VISIBLE_DEVICES=$gid fairseq-generate $DATA \
                    --finetune-data $DATA \
                    -t ${target_lg} -s ${source_lg} \
                    --finetune-langs ${source_lg}-${target_lg} \
                    --path $model \
                    --task $task \
                    --skip-invalid-size-inputs-valid-test \
                    --gen-subset $SPLIT \
                    --lyrics-dict $lyrics_dict \
                    --align-rests \
                    --align-notes \
                    --distance-reward \
                    --notes-weight $notes_weight \
                    --durations-weight $durations_weight \
                    --rests-weight $rests_weight \
                    --shape-weight $shape_weight \
                    --min-align-prob $min_align_prob \
                    --langs en,zh \
                    --placeholder $placeholder \
                    --domains LYRICS,WMT \
                    --ft-domain $ft_domain \
                    --add-lang-token \
                    --prefix-size $prefix_size \
                    --mono-ratio 0.0 \
                    --bpe 'fastbpe' \
                    --bpe-codes $BPE_CODES \
                    --sacrebleu  \
                    --remove-bpe '@@' \
                    --max-sentences 16 \
                    --beam $beam_size \
                    --no-progress-bar $suffix > ${SAVE_PATH}/${source_lg}-${target_lg}-${SPLIT}

cat ${SAVE_PATH}/${source_lg}-${target_lg}-${SPLIT} | grep -P "^H" |sort -V |cut -f 3- | sed "s/\[$ft_domain\]//g" > ${SAVE_PATH}/$target_lg.$SPLIT.hyp.org
cat ${SAVE_PATH}/${source_lg}-${target_lg}-${SPLIT} | grep -P "^T" |sort -V |cut -f 2- | sed "s/\[$ft_domain\]//g" > ${SAVE_PATH}/$target_lg.$SPLIT.ref
cat ${SAVE_PATH}/${source_lg}-${target_lg}-${SPLIT} | grep -P "^S" |sort -V |cut -f 2- | sed "s/\[$ft_domain\]//g" > ${SAVE_PATH}/$source_lg.$SPLIT.src

cp $DATA/${SPLIT}_tgt_len ${SAVE_PATH}/${SPLIT}_tgt_len

if [[ $suffix == "" ]]
then
    cp ${SAVE_PATH}/$target_lg.$SPLIT.hyp.org ${SAVE_PATH}/$target_lg.$SPLIT.hyp
else
    python ./restore_group.py $DATA/${SPLIT}.notes.group.pitches_dur ${SAVE_PATH}/$target_lg.$SPLIT.hyp.org ${SAVE_PATH}/$target_lg.$SPLIT.hyp
fi
