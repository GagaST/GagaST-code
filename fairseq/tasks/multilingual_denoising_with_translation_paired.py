# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import glob
import torch
import itertools

import numpy as np
from fairseq import metrics, options, utils
from fairseq import search


from fairseq.data import (
    encoders,
    indexed_dataset,
    AppendTokenDataset,
    ConcatDataset,
    StripTokenDataset,
    TruncateDataset,
    XDAEDenoisingPairDataset,
    Dictionary,
    PrependTokenDataset,
    ResamplingDataset,
    SortDataset,
    TokenBlockDataset,
    MultiLanguagePairDataset,
    data_utils,
)
from .denoising import DenoisingTask
from .translation import TranslationTask
from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task


logger = logging.getLogger(__name__)


def load_multi_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    add_language_token=False,
    use_language_eos=False,
    domain=None,
    use_domain_eos=False,
    common_eos=None,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    def replace_eos(dataset, dictionary, eos_token):
        dataset = StripTokenDataset(dataset, dictionary.eos())
        eos_index = dictionary.index("[{}]".format(eos_token))
        return AppendTokenDataset(dataset, eos_index)

    src_datasets = []
    tgt_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - tag_num - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    # add tags
    if domain is not None:
        src_dataset = PrependTokenDataset(src_dataset, src_dict.index("[{}]".format(domain)))
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, src_dict.index("[{}]".format(domain)))
    if add_language_token:
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(
                tgt_dataset, tgt_dict.index('[2{}]'.format(tgt))
            )
        src_dataset = PrependTokenDataset(
                src_dataset, tgt_dict.index('[2{}]'.format(tgt)) 
            )
    # use either common eos, domain tag, language tag as eos
    if common_eos is not None:
        src_dataset = replace_eos(src_dataset, src_dict, common_eos) 
        if tgt_dataset is not None:
            tgt_dataset = replace_eos(tgt_dataset, tgt_dict, common_eos)
    elif domain is not None and use_domain_eos:
        src_dataset = replace_eos(src_dataset, src_dict, domain) 
        if tgt_dataset is not None:
            tgt_dataset = replace_eos(tgt_dataset, tgt_dict, domain)
    elif use_language_eos:
        src_dataset = replace_eos(src_dataset, src_dict, src) 
        if tgt_dataset is not None:
            tgt_dataset = replace_eos(tgt_dataset, tgt_dict, tgt)

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return MultiLanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )

@register_task('xdae_multilingual_translation_pair')
class XDAEMultilingualTranslationPairTask(DenoisingTask):
    @staticmethod
    def add_args(parser):
        DenoisingTask.add_args(parser)
        # for pretrain
        parser.add_argument(
            "--multilang-sampling-alpha",
            type=float,
            default=1.0,
            help="smoothing alpha for sample ratios across multiple datasets",
        )
        parser.add_argument("--downsample-by-min", default=False, action="store_true",
                            help="Downsample all large dataset by the length of smallest dataset")
        parser.add_argument("--add-lang-token", default=False, action="store_true")
        parser.add_argument("--use-lang-eos", default=False, action="store_true")
        parser.add_argument("--with-len", default=False, action="store_true")
        parser.add_argument('--prepend-bos', default=False, action='store_true')

        parser.add_argument('--placeholder', type=int,
                            help="placeholder for more special ids such as language ids",
                            default=-1)
        parser.add_argument("--add-tgt-len-tags", type=int, default=0,
                            help="number of length tags to add")
        parser.add_argument('--word-shuffle', type=float, default=0,
                            help="Randomly shuffle input words (0 to disable)")
        parser.add_argument("--word-dropout", type=float, default=0,
                            help="Randomly dropout input words (0 to disable)")
        parser.add_argument("--word-blank", type=float, default=0,
                            help="Randomly blank input words (0 to disable)")
        

        parser.add_argument('--sampled-data', default=False, action='store_true')
        parser.add_argument(
            "--langs", type=str, help="language ids we are considering", default=None
        )
        parser.add_argument(
            "--no-whole-word-mask-langs",
            type=str,
            default="",
            metavar="N",
            help="languages without spacing between words dont support whole word masking",
        )
        parser.add_argument('--finetune-langs', type=str, 
                            help="language pairs to finetune',', for example, 'en-zh,zh-en'", 
                            default=None)
        parser.add_argument('--finetune-data', type=str, 
                            help="finetuning data path", 
                            default=None)
        parser.add_argument('--common-eos', type=str, 
                            help="common end of sentence tag for all tasks/langs", 
                            default=None)
        parser.add_argument('--domains', type=str, 
                            help="domains to pretrain ',', for example, 'LYRICS,WMT'", 
                            default=None)
        parser.add_argument('--mono-langs', type=str, 
                            help="monolingual languages used in pretraining, separated wiht ',', for example, 'en,fr,de'", 
                            default=None)
        parser.add_argument('--para-langs', type=str, 
                            help="parallel langagues, for example, 'en-zh,jp-zh'",
                            default=None)
        parser.add_argument('--mono-domain', type=str, 
                            help="domain of monolingual data", 
                            default=None)
        parser.add_argument('--para-domain', type=str, 
                            help="domain of parallel data", 
                            default=None)
        parser.add_argument('--ft-domain', type=str, 
                            help="domain of fintuning data", 
                            default=None)
        parser.add_argument("--use-domain-eos", action="store_true",
                            help="use domain tag as end of sentence",
                            default=False)
        parser.add_argument("--mono-ratio", type=float,
                            help="Percentage of monolingual data",
                            default=0.5)

        # for generation
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')

        ## options for reporting BLEU during validation
        #parser.add_argument('--eval-bleu', action='store_true',
                            #help='evaluation with BLEU scores')
        #parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            #help='detokenizer before computing BLEU (e.g., "moses"); '
                                 #'required if using --eval-bleu; use "space" to '
                                 #'disable detokenization; see fairseq.data.encoders '
                                 #'for other options')
        #parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            #help='args for building the tokenizer, if needed')
        #parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            #help='if setting, we compute tokenized BLEU instead of sacrebleu')
        #parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            #help='remove BPE before computing BLEU')
        #parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            #help='generation args for BLUE scoring, '
                                 #'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        #parser.add_argument('--eval-bleu-print-samples', action='store_true',
        #                    help='print sample generations during validation')
        

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task."""
        paths = args.data.split(":")
        assert len(paths) > 0
        
        if args.langs is None:
            if args.sampled_data:
                languages = list(cls.get_languages(cls, paths[0]))
            else:
                languages = sorted([
                    name for name in os.listdir(paths[0])
                    if os.path.isdir(os.path.join(paths[0], name))
                ])
        else:
            languages = args.langs.split(",")

        dict_path = paths[1] if len(paths) == 2 else paths[0]
        if os.path.exists(os.path.join(dict_path, "dict.txt")):
            dictionary = Dictionary.load(os.path.join(dict_path, "dict.txt"))
        else:
            dictionary = Dictionary.load(os.path.join(dict_path, f"dict.{languages[0]}.txt"))

        domains = args.domains.split(",") if args.domains is not None else None
        assert (args.mono_domain is None) or (args.mono_domain in domains)
        assert (args.para_domain is None) or (args.para_domain in domains) 

        dictionary.add_symbol('<mask>')
        if args.add_lang_token:
            if args.common_eos is not None:
                dictionary.add_symbol('[{}]'.format(args.common_eos))
            if domains is not None:
                for d in domains:
                    dictionary.add_symbol(f"[{d}]")
            for lang in languages:
                dictionary.add_symbol('[2{}]'.format(lang))
            if args.add_tgt_len_tags > 0:
                for i in range(args.add_tgt_len_tags):
                    dictionary.add_symbol('[LEN{}]'.format(i+1))
            if args.placeholder > 0:
                for i in range(args.placeholder):
                    dictionary.add_symbol('[placeholder{}]'.format(i))
            

        logger.info("dictionary: {} types".format(len(dictionary)))
        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False

        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        return cls(args, dictionary)

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.dictionary = dictionary
        self.src_dict = dictionary
        self.tgt_dict = dictionary
        self.seed = args.seed

        # add mask token
        self.mask_idx = self.dictionary.index('<mask>')
        self.langs = args.langs
        self.args = args
        self.path_cache = {}
        self.para_langs = None if args.para_langs is None else args.para_langs.split(",")
        self.ft_langs = None if args.finetune_langs is None else args.finetune_langs.split(",") 
        self.mono_langs = args.mono_langs.split(",") if args.mono_langs is not None else None

    def _get_sample_prob(self, dataset_lens):
        """
        Get smoothed sampling porbability by languages. This helps low resource
        languages by upsampling them.
        """
        
        prob = dataset_lens / dataset_lens.sum()
        smoothed_prob = prob ** self.args.multilang_sampling_alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        return smoothed_prob

    def get_languages(self, data_folder):
        files = [path for path in os.listdir(data_folder)]
        lgs = set([x.split('.')[-2] for x in files])
        return lgs

    def get_dataset_path(self, split, data_folder, epoch, lgs=None, is_pair=False):
        if data_folder in self.path_cache:
            files = self.path_cache[data_folder]
        else:
            files = [path for path in os.listdir(data_folder)]
            # remove this to speed up
            # if os.path.isfile(os.path.join(data_folder, path))
            self.path_cache[data_folder] = files

        files = [path for path in files if(split in path) and (".bin" in path)]  

        if lgs is None:
            lgs = set([x.split('.')[-2] for x in files])

        paths = {} 
        for lg_index, lg in enumerate(lgs):
            if is_pair:
                pair = lg.split('-')
                split_count = len([path for path in files if ".{0}.{1}.bin".format(lg, pair[0]) in path])
            else:
                split_count = len([path for path in files if ".{0}.bin".format(lg) in path])
            big_step = epoch // split_count
            small_step = epoch % split_count
            with data_utils.numpy_seed((self.args.seed + big_step) * 100 + lg_index):
                shuffle = np.random.permutation(split_count)
                index = shuffle[small_step]
                path = os.path.join(data_folder, "{0}.{1}.{2}".format(split, index, lg))
                paths[lg] = path
        return paths

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0
        # pretrained dataset path
        mono_ratio = self.args.mono_ratio
        para_path = ""
        para_dataset = None
        lang_splits = [split]
        if split == getattr(self.args, "train_subset", "train"):
            if mono_ratio > 1e-5 : 
                data_path = paths[0]
                split_path = os.path.join(data_path, split)

                sampled = self.args.sampled_data
                languages = self.mono_langs

                if sampled:
                    all_lg_path = self.get_dataset_path(split, data_path, epoch, languages)
                    if languages is None:
                        languages = list(all_lg_path.keys())
                else:
                    all_lg_path = None
                    if languages is None:
                        languages = sorted([
                            lang for lang in os.listdir(data_path)
                            if os.path.isdir(os.path.join(data_path, lang))
                        ])
                    else:
                        for lang in languages:
                            assert os.path.exists(os.path.join(data_path, lang)), "all the languages must exist"

                logger.info("Training on {0} languages: {1}".format(len(languages), languages))
                logger.info("Language to id mapping: {}".format({
                        lang: ids for ids, lang in enumerate(languages)
                    })
                )

                mask_whole_words = get_whole_word_mask(self.args, self.dictionary)
                language_without_segmentations = self.args.no_whole_word_mask_langs.split(",")
                lang_datasets = []

                for language in languages:
                    tag_num = int(self.args.with_len)
                    split_path = os.path.join(data_path, language, split) if all_lg_path is None else all_lg_path[language]
                    dataset = data_utils.load_indexed_dataset(
                        split_path,
                        self.source_dictionary,
                        self.args.dataset_impl,
                        combine=combine,
                    )
                    if dataset is None:
                        raise FileNotFoundError("Dataset not found: {} ({})".format(split, split_path))
                    
                    if self.args.common_eos:
                        end_token = self.source_dictionary.index('[{}]'.format(self.args.common_eos))
                    elif self.args.use_domain_eos:
                        end_token = self.source_dictionary.index('[{}]'.format(self.args.mono_domain))
                    elif self.args.use_lang_eos:
                        end_token = self.source_dictionary.index('[2{}]'.format(language)) 
                    else:
                        end_token = self.source_dictionary.eos()
                    
                    # create continuous blocks of tokens
                    strip_length = 2 if self.args.mono_domain is None else 3
                    dataset = TokenBlockDataset(
                        dataset,
                        dataset.sizes,
                        self.args.tokens_per_sample - strip_length,  # one less for <s>
                        pad=self.source_dictionary.pad(),
                        eos=end_token,
                        break_mode=self.args.sample_break_mode,
                    )
                    logger.info('loaded {} blocks from: {}'.format(len(dataset), split_path))

                    # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
                    bos_idx = None
                    if self.args.prepend_bos:
                        bos_idx = self.source_dictionary.bos()
                        dataset = PrependTokenDataset(dataset, bos_idx)

                    if self.args.mono_domain is not None:
                        tag_num += 1
                        bos_idx = self.source_dictionary.index('[{}]'.format(self.args.mono_domain))
                        dataset = PrependTokenDataset(dataset, bos_idx)

                    if self.args.add_lang_token:
                        tag_num += 1
                        bos_idx = self.source_dictionary.index('[2{}]'.format(language))
                        dataset = PrependTokenDataset(dataset, bos_idx)
                    # replace end token
                    dataset = StripTokenDataset(dataset, self.source_dictionary.eos())
                    dataset = AppendTokenDataset(dataset, end_token)
                    lang_mask_whole_words = mask_whole_words if language not in language_without_segmentations else None

                    lang_dataset = XDAEDenoisingPairDataset(
                        dataset,
                        dataset.sizes,
                        self.dictionary,
                        self.mask_idx,
                        lang_mask_whole_words,
                        shuffle=self.args.shuffle_instance,
                        seed=self.seed,
                        args=self.args,
                        tag_num=tag_num,
                        eos=end_token,
                    )
                    lang_datasets.append(lang_dataset)

                dataset_lengths = np.array(
                    [len(d) for d in lang_datasets],
                    dtype=float,
                )
                logger.info(
                    'loaded total {} blocks for all languages'.format(
                        dataset_lengths.sum(),
                    )
                )
                if not self.args.sampled_data:
                    #For train subset, additionally up or down sample languages.
                    if self.args.downsample_by_min:
                        min_len = min(dataset_lengths)
                        size_ratio = min_len / dataset_lengths
                    else:
                        sample_probs = self._get_sample_prob(dataset_lengths)
                        logger.info("Sample probability by language: {}".format({
                                lang: "{0:.4f}".format(sample_probs[id])
                                for id, lang in enumerate(languages)
                            })
                        )
                        size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
                    logger.info("Up/Down Sampling ratio by language: {}".format({
                            lang: "{0:.2f}".format(size_ratio[id])
                            for id, lang in enumerate(languages)
                        })
                    )

                    resampled_lang_datasets = [
                        ResamplingDataset(
                            lang_datasets[i],
                            size_ratio=size_ratio[i],
                            seed=self.args.seed,
                            epoch=epoch,
                            replace=size_ratio[i] >= 1.0,
                        )
                        for i, d in enumerate(lang_datasets)
                    ]
                    mono_dataset = ConcatDataset(
                        resampled_lang_datasets,
                        )
                else:
                    mono_dataset = ConcatDataset(
                        lang_datasets,
                    )

            # start loading parallel dataset 
            para_path = paths[1] if self.args.mono_ratio > 1e-5 else paths[0]
            para_datasets = []
            for pair in self.para_langs:
                src, tgt = pair.split("-") 
                lang_dataset = load_multi_langpair_dataset(
                    para_path,
                    split,
                    src,
                    self.source_dictionary,
                    tgt,
                    self.target_dictionary,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=getattr(self.args, 'max_source_positions', 512),
                    max_target_positions=getattr(self.args, 'max_target_positions', 512),
                    prepend_bos=getattr(self.args, 'preprend_bos', False),
                    add_language_token=self.args.add_lang_token,
                    use_language_eos=self.args.use_lang_eos,
                    domain=self.args.para_domain,
                    use_domain_eos=self.args.use_domain_eos,
                    common_eos=self.args.common_eos,
                    )
                para_datasets.append(lang_dataset)

            if len(para_datasets) > 1:
                dataset_lengths = np.array([len(d) for d in para_datasets], dtype=float)

                sample_probs = self._get_sample_prob(dataset_lengths)
                logger.info("Sample probability by language pair: {}".format({
                        pair: "{0:.4f}".format(sample_probs[id])
                        for id, pair in enumerate(self.para_langs)
                    })
                )
                size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
                logger.info("Up/Down Sampling ratio by language for finetuning: {}".format({
                        pair: "{0:.2f}".format(size_ratio[id])
                        for id, pair in enumerate(self.para_langs)
                    })
                )
            
                resampled_lang_datasets = [
                    ResamplingDataset(
                        para_datasets[i],
                        size_ratio=size_ratio[i],
                        seed=self.args.seed,
                        epoch=epoch,
                        replace=size_ratio[i] >= 1.0,
                    )
                    for i, d in enumerate(para_datasets)
                ]
                para_dataset = ConcatDataset(
                    resampled_lang_datasets,
                    )
            else:
                para_dataset = para_datasets[0]

            if mono_ratio > 1e-5:
                mono_len, para_len = len(mono_dataset), len(para_dataset)
                if mono_len > para_len:
                    ratio = float(para_len/mono_len)*mono_ratio/(1.0 - mono_ratio)
                    logger.info("Down sampling probability for monolingual data: {}".format(ratio))
                
                    mono_dataset = ResamplingDataset(
                                                mono_dataset,
                                                size_ratio=ratio,
                                                seed=self.args.seed,
                                                epoch=epoch,
                                                replace=ratio >= 1.0,
                                            )
                else:
                    ratio = float(mono_len/para_len)*(1.0 - mono_ratio)/mono_ratio
                    logger.info("Down sampling probability for parallel data: {}".format(ratio))
                
                    para_dataset = ResamplingDataset(
                                                para_dataset,
                                                size_ratio=ratio,
                                                seed=self.args.seed,
                                                epoch=epoch,
                                                replace=ratio >= 1.0,
                                            )
               
                para_dataset = ConcatDataset(
                        [para_dataset, mono_dataset]
                    )

        ft_path = self.args.finetune_data 
        ft_datasets = []
        if not (split == getattr(self.args, "train_subset", "train") and (ft_path == para_path)):
            for pair in self.ft_langs:
                src, tgt = pair.split("-") 
                lang_dataset = load_multi_langpair_dataset(
                    ft_path,
                    split,
                    src,
                    self.source_dictionary,
                    tgt,
                    self.target_dictionary,
                    combine=combine,
                    dataset_impl=self.args.dataset_impl,
                    upsample_primary=self.args.upsample_primary,
                    left_pad_source=self.args.left_pad_source,
                    left_pad_target=self.args.left_pad_target,
                    max_source_positions=getattr(self.args, 'max_source_positions', 512),
                    max_target_positions=getattr(self.args, 'max_target_positions', 512),
                    prepend_bos=getattr(self.args, 'preprend_bos', False),
                    add_language_token=self.args.add_lang_token,
                    use_language_eos=self.args.use_lang_eos,
                    domain=self.args.ft_domain,
                    use_domain_eos=self.args.use_domain_eos,
                    common_eos=self.args.common_eos,
                    )
                ft_datasets.append(lang_dataset)

        if split == getattr(self.args, "train_subset", "train"):
            if len(ft_datasets) > 1:
                dataset_lengths = np.array([len(d) for d in ft_datasets], dtype=float)

                sample_probs = self._get_sample_prob(dataset_lengths)
                logger.info("Sample probability by language pair: {}".format({
                        pair: "{0:.4f}".format(sample_probs[id])
                        for id, pair in enumerate(self.ft_langs)
                    })
                )
                size_ratio = (sample_probs * dataset_lengths.sum()) / dataset_lengths
                logger.info("Up/Down Sampling ratio by language for finetuning: {}".format({
                        pair: "{0:.2f}".format(size_ratio[id])
                        for id, pair in enumerate(self.ft_langs)
                    })
                )
            
                resampled_lang_datasets = [
                    ResamplingDataset(
                        ft_datasets[i],
                        size_ratio=size_ratio[i],
                        seed=self.args.seed,
                        epoch=epoch,
                        replace=size_ratio[i] >= 1.0,
                    )
                    for i, d in enumerate(ft_datasets)
                ]
                ft_dataset = ConcatDataset(
                    resampled_lang_datasets,
                    )
            else:
                ft_dataset = ft_datasets[0] if len(ft_datasets) > 0 else None
        else:
            ft_dataset = ConcatDataset(ft_datasets)
            domain_name = "_{}".format(self.args.ft_domain) if self.args.ft_domain is not None else "" 
            for lang_id, lang_dataset in enumerate(ft_datasets):
                split_name = split + "_" + self.ft_langs[lang_id] + domain_name
                lang_splits.append(split_name)
                self.datasets[split_name] = lang_dataset

            if hasattr(self.args, "valid_subset"):
                if split in self.args.valid_subset:
                    self.args.valid_subset = self.args.valid_subset.replace(
                        split, ','.join(lang_splits)
                   )
        if para_dataset is None:
            assert ft_dataset is not None, "must have at least some dataset"
            para_dataset = ft_dataset
        elif ft_dataset is not None:
            para_dataset = ConcatDataset(
                    [para_dataset, ft_dataset]
                )

        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(para_dataset))

        self.datasets[split] = SortDataset(
            para_dataset,
            sort_order=[
                shuffle,
                para_dataset.sizes,
            ],
        )   

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            bos_idx = None
            if self.args.prepend_bos:
                bos_idx = self.dictionary.bos()
            if self.args.ft_domain is not None:
                bos_idx = self.dictionary.index('[{}]'.format(self.args.ft_domain))
            if self.args.add_lang_token:
                bos_idx = self.dictionary.index('[2{}]'.format(self.args.target_lang))

            return generator.generate(models, sample, prefix_tokens=prefix_tokens, bos_token=bos_idx)

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None
    ):
        if self.args.common_eos:
            eos = self.source_dictionary.index('[{}]'.format(self.args.common_eos))
        elif self.args.use_domain_eos:
            eos = self.source_dictionary.index('[{}]'.format(self.args.ft_domain))
        elif self.args.use_lang_eos:
            eos = self.source_dictionary.index('[2{}]'.format(args.target_lang)) 
        else:
            eos = self.source_dictionary.eos()

        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                compute_alignment=getattr(args, "print_alignment", False),
            )

        from fairseq.sequence_generator_with_prefix import (
            SequenceGenerator,
            SequenceGeneratorWithAlignment,
        )
        try:
            from fairseq.fb_sequence_generator import FBSequenceGenerator
        except ModuleNotFoundError:
            pass

        # Choose search strategy. Defaults to Beam Search.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
        diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
        match_source_len = getattr(args, "match_source_len", False)
        diversity_rate = getattr(args, "diversity_rate", -1)
        constrained = getattr(args, "constraints", False)
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
        if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
        ):
            raise ValueError("Provided Search parameters are mutually exclusive.")
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            search_strategy = search.Sampling(
                self.target_dictionary, sampling_topk, sampling_topp
            )
        elif diverse_beam_groups > 0:
            search_strategy = search.DiverseBeamSearch(
                self.target_dictionary, diverse_beam_groups, diverse_beam_strength
            )
        elif match_source_len:
            # this is useful for tagging applications where the output
            # length should match the input length, so we hardcode the
            # length constraints for simplicity
            search_strategy = search.LengthConstrainedBeamSearch(
                self.target_dictionary,
                min_len_a=1,
                min_len_b=0,
                max_len_a=1,
                max_len_b=0,
            )
        elif diversity_rate > -1:
            search_strategy = search.DiverseSiblingsSearch(
                self.target_dictionary, diversity_rate
            )
        elif constrained:
            search_strategy = search.LexicallyConstrainedBeamSearch(
                self.target_dictionary, args.constraints
            )
        elif prefix_allowed_tokens_fn:
            search_strategy = search.PrefixConstrainedBeamSearch(
                self.target_dictionary, prefix_allowed_tokens_fn
            )
        else:
            search_strategy = search.BeamSearch(self.target_dictionary)

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            if getattr(args, "print_alignment", False):
                seq_gen_cls = SequenceGeneratorWithAlignment
                extra_gen_cls_kwargs["print_alignment"] = args.print_alignment
            elif getattr(args, "fb_seq_gen", False):
                seq_gen_cls = FBSequenceGenerator
            else:
                seq_gen_cls = SequenceGenerator

        return seq_gen_cls(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            eos=eos,
            **extra_gen_cls_kwargs,
        )

