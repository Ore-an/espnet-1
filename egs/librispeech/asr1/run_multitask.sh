#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4       # start from -1 if you need to start from data download
stop_stage=100
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu)
nj=32
dnj=30
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume= # Resume the training from snapshot

# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
train_config=conf/train_auxph.yaml # current default recipe requires 4 gpus.
                             # if you do not have 4 gpus, please reconfigure the `batch-bins` and `accum-grad` parameters in config.
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=0               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
datadir=/disk/scratch4/acarmant/kaldi/egs/librispeech/s5/data/LibriSpeech

# base url for downloads.
data_url=www.openslr.org/resources/12

# bpemode (unigram or bpe)
nbpe=632
bpemode=bpe5000_auxmetaphoneme
lexicon=unilex/unilex_baseform

# exp tag
tag="auxmph" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_960
train_dev=dev
recog_set=""

for acc in african australia bermuda canada england hongkong indian ireland malaysia newzealand philippines scotland singapore us; do
    recog_set+="commonvoice/$acc/test "
done

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in dev_clean test_clean dev_other test_other train_clean_100 train_clean_360 train_other_500; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj ${nj} --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    utils/combine_data.sh --extra_files utt2num_frames data/${train_set}_org data/train_clean_100 data/train_clean_360 data/train_other_500
    utils/combine_data.sh --extra_files utt2num_frames data/${train_dev}_org data/dev_clean data/dev_other

    # remove utt having more than 3000 frames
    # remove utt having more than 400 characters
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}
    remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_dev}_org data/${train_dev}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_char/train_960_unigram5000_units.txt
bpemodel=data/lang_char/train_960_unigram5000

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
        data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model \
            data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

dumpdir+=${tag}
feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
lang_dir=data/lang_mph
aux_dict=${lang_dir}/${train_set}_${bpemode}${nbpe}_units.txt

echo "dictionary: ${aux_dict}"
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 3: Secondary task dictionary and Json Data Preparation"
    mkdir -p ${lang_dir}
    cut -d ':' -f 1  ${lexicon} | tr '[:lower:]' '[:upper:]' > ${lang_dir}/wd
    cut -d ':' -f 4  ${lexicon} | tr -d '{*}.=<>[]:~\-$' | column -t | tr -s ' ' | \
        perl -pe 'sub c{$s=shift;$s=~s/ //g;$s}s/\(.*?\)/c$&/ge' | sed -E 's:(^| )[1/]( |$)::g' > ${lang_dir}/pr    # removing symbols for stress and etc, removing superfluous whitespace, compressing alternative metaph in parentheses
    paste -d ' ' ${lang_dir}/{wd,pr} > ${lang_dir}/lexicon
    rm ${lang_dir}/{wd,pr}
    echo "<unk> 1" > ${aux_dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -d ' ' -f 2- ${lang_dir}/lexicon | tr ' ' '\n' | sort -u | awk '{ print $0, NR + 1}' >> ${aux_dict}
    wc -l ${aux_dict}
    
    mkdir -p ${lang_dir}/g2p
    echo "Making g2p model"
    g2p.py --train ${lang_dir}/lexicon --devel 5% --write-model ${lang_dir}/g2p/model-1 > ${lang_dir}/g2p/model-1.log 2>&1
    g2p.py --model ${lang_dir}/g2p/model-1 --ramp-up --train ${lang_dir}/lexicon --devel 5% --write-model ${lang_dir}/g2p/model-2 > ${lang_dir}/g2p/model-2.log 2>&1
    g2p.py --model ${lang_dir}/g2p/model-2 --ramp-up --train ${lang_dir}/lexicon --devel 5% --write-model ${lang_dir}/g2p/model-3 > ${lang_dir}/g2p/model-3.log 2>&1
    g2p.py --model ${lang_dir}/g2p/model-3 --ramp-up --train ${lang_dir}/lexicon --devel 5% --write-model ${lang_dir}/g2p/model-4 > ${lang_dir}/g2p/model-4.log 2>&1
    g2p.py --model ${lang_dir}/g2p/model-4 --ramp-up --train ${lang_dir}/lexicon --devel 5% --write-model ${lang_dir}/g2p/model-5 > ${lang_dir}/g2p/model-5.log 2>&1

    cut -d ' ' -f 2- data/${train_set}/text | tr ' ' '\n' | sort -u > ${lang_dir}/words.txt
    g2p.py --model ${lang_dir}/g2p/model-5 --apply ${lang_dir}/words.txt > ${lang_dir}/lexicon_expansion
    cat ${lang_dir}/lexicon{,_expansion} | sort -u | awk '!seen[$1]++' > ${lang_dir}/expanded_lexicon
    # make json labels  
    for exdump in dump/*; do
    	ds=$(basename ${exdump})
    	mkdir -p ${dumpdir}/${ds}/delta${do_delta}
    	cp dump/${ds}/delta${do_delta}/feats.scp ${dumpdir}/${ds}/delta${do_delta}/
    done
    data2json_aux.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model --lexicon ${lang_dir}/expanded_lexicon \
        data/${train_set} ${dict} ${aux_dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
    data2json_aux.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model --lexicon ${lang_dir}/expanded_lexicon \
        data/${train_dev} ${dict} ${aux_dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json

    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json_aux.sh --feat ${feat_recog_dir}/feats.scp --bpecode ${bpemodel}.model --lexicon ${lang_dir}/expanded_lexicon \
            data/${rtask} ${dict} ${aux_dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json
    done
fi

# You can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=${backend}_${lmtag}_${bpemode}${nbpe}_ngpu${ngpu}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
        expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train_auxtask.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
	--aux_dict ${aux_dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]] || \
       [[ $(get_yaml.py ${train_config} model-module) = *conformer* ]]; then
        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
	if [ ! -f ${expdir}/results/${recog_model} ]; then
            average_checkpoints.py \
		${opt} \
		--backend ${backend} \
		--snapshots ${expdir}/results/snapshot.ep.* \
		--out ${expdir}/results/${recog_model} \
		--num ${n_average}
	fi

        # Average LM models
        if [ ${lm_n_average} -eq 0 ]; then
            lang_model=rnnlm.model.best
        else
            if ${use_lm_valbest_average}; then
                lang_model=rnnlm.val${lm_n_average}.avg.best
                opt="--log ${lmexpdir}/log"
            else
                lang_model=rnnlm.last${lm_n_average}.avg.best
                opt="--log"
            fi
            average_checkpoints.py \
                ${opt} \
                --backend ${backend} \
                --snapshots ${lmexpdir}/snapshot.ep.* \
                --out ${lmexpdir}/${lang_model} \
                --num ${lm_n_average}
        fi
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${dnj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${dnj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 1 \
            --recog-json ${feat_recog_dir}/split${dnj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm exp/pytorch_lm_unigram5000_ngpu4//${lang_model} \
            --api v2

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    done
    echo "Finished"
fi
