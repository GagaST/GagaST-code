# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


VOCAB_DIR=$1     # path/to/vocab
DATA=$2          # path/to/data

DICT=$VOCAB_DIR/vocab

DATA_BPE=$DATA/bpe
DATA_BIN=$DATA/bin
DATA_REF=$DATA/ref

mkdir -p $DATA_BIN
mkdir -p $DATA_REF

# Save references

#for lg in en es fr de ru; do
    #cp ${DATA}/xglue.ntg.$lg.tgt.dev ${DATA_REF}/$lg.tgt.valid 
#done


# Binarize

fairseq-preprocess \
    --source-lang en \
    --target-lang zh \
    --trainpref $DATA_BPE/en-zh.train.bpe \
    --validpref $DATA_BPE/en-zh.valid.bpe \
    --testpref $DATA_BPE/en-zh.test.bpe \
    --destdir $DATA_BIN/$lg \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 96 

echo "Done!"
