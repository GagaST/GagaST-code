gid=$1
EXP=$2 #Pretrain_all_musescore_filtered_single_tag_lr3e-4_m30_r0.5_mtoken4096_upf2_M4_len_no_puncs
split="test"
bs=10
tp="line"
lp=$3
min_prob=0.01 #0.05 # 0.1
notes_w=$4
shape_w=$5
dur_w=$6
rest_w=$7

suffix=""
if [[ $lp == "syllable" ]]; then
    suffix="with_group"
fi
echo $suffix
                            
bash ./generate_single_constrained.sh $gid en zh /data/data/musescore/loose/data_split/${tp}/${lp}/no_rest/bin/  \
     /data/data/processed/vocab/wmt_lyrics/codes LYRICS 0 $split \
     /data/projects/output/$EXP/checkpoint_last.pt $bs \
     /data/data/lyrics/jieba_lyrics.vocab \
      $notes_w $shape_w $dur_w $rest_w $min_prob \
     /data/projects/output/$EXP/checkpoint_last/constrained_distance_reward/${split}/${tp}/${lp}/${split}_beam-${bs}_notes-${notes_w}_shape-${shape_w}_duration-${dur_w}_rest-${rest_w}_minProb-${min_prob} $suffix 
