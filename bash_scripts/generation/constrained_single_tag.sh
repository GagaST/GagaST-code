gid=$1
EXP=Pretrain_all_musescore_filtered_single_tag_lr3e-4_m30_r0.5_mtoken4096_upf2_M4_len_no_puncs
NGPU=8
for split in valid; do
    for tp in line seg; do
        for bs in 10
        do
            for lp in syllable
            do
                for rest in no_rest; do
                    for min_prob in 0.01 #0.05 # 0.1
                    do
                        for notes_w in 0.0 0.5 1.0
                        do
                            for shape_w in 0.0 0.5 1.0 
                            do 
                                for dur_w in 0.0 #0.5 1.0 1.5 2.0 3.0
                                do
                                    for rest_w  in 0.0 0.5 1.0 1.5
                                    do
                                        suffix=""
                                        if [[ $lp == "syllable" ]]; then
                                            suffix="with_group"
                                        fi
                                        echo $suffix
                                        bash ./generate_single_constrained.sh $gid en zh /data/data/musescore/loose/data_split/${tp}/${lp}/${rest}/bin/  \
                                             /data/data/processed/vocab/wmt_lyrics/codes LYRICS 0 $split \
                                             /data/projects/output/$EXP/checkpoint_last.pt $bs \
                                             /data/data/lyrics/jieba_lyrics.vocab \
                                              $notes_w $shape_w $dur_w $rest_w $min_prob \
                                             /data/projects/output/$EXP/checkpoint_last/constrained_no_cut_len/${split}/${tp}/${lp}/${rest}/${split}_beam-${bs}_notes-${notes_w}_shape-${shape_w}_duration-${dur_w}_rest-${rest_w}_minProb-${min_prob} $suffix & 
                                        gid=$(($gid+1))
                                        if [ $(($gid%$NGPU)) = 0 ]; then
                                            wait
                                            gid=0
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

for split in valid; do
    for tp in line seg; do
        for bs in 10
        do
            for lp in notes 
            do
                for rest in no_rest; do
                    for min_prob in 0.01 #0.05 # 0.1
                    do
                        for notes_w in 0.0 0.5 1.0
                        do
                            for shape_w in 0.0 #0.5 1.0 
                            do 
                                for dur_w in 0.0 #0.5 1.0 1.5 2.0 3.0
                                do
                                    for rest_w  in 0.0 0.5 1.0 1.5
                                    do
                                        suffix=""
                                        if [[ $lp == "syllable" ]]; then
                                            suffix="with_group"
                                        fi
                                        echo $suffix
                                        bash ./generate_single_constrained.sh $gid en zh /data/data/musescore/loose/data_split/${tp}/${lp}/${rest}/bin/  \
                                             /data/data/processed/vocab/wmt_lyrics/codes LYRICS 0 $split \
                                             /data/projects/output/$EXP/checkpoint_last.pt $bs \
                                             /data/data/lyrics/jieba_lyrics.vocab \
                                              $notes_w $shape_w $dur_w $rest_w $min_prob \
                                             /data/projects/output/$EXP/checkpoint_last/constrained_no_cut_len/${split}/${tp}/${lp}/${rest}/${split}_beam-${bs}_notes-${notes_w}_shape-${shape_w}_duration-${dur_w}_rest-${rest_w}_minProb-${min_prob} $suffix & 
                                        gid=$(($gid+1))
                                        if [ $(($gid%$NGPU)) = 0 ]; then
                                            wait
                                            gid=0
                                        fi
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
