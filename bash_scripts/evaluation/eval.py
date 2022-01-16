from pypinyin import lazy_pinyin, Style
import re
import math
import jieba
import glob
import os
import sys
import pandas as pd
import numpy as np
import subprocess
import getopt
from scipy.optimize import leastsq


def preprocess_lyrics(lyrics):
    num_mappings = "零一二三四五六七八九"
    if bool(re.search(r"[a-zA-Z]", lyrics.replace("[punc]", "").replace("[SEP]", ""))):  # 含有英文
        return None
    else:
        lyrics = lyrics.replace(" ", "")
        for i in range(10):
            lyrics = lyrics.replace(str(i), num_mappings[i])
        return lyrics


def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c

def error(params, x, y):
    return func(params, x) - y

def slovePara(X, Y):
    p0 = [10, 10, 10]
    Para = leastsq(error, p0, args=(X, Y))
    return Para

def notes2shape(notes):
    X = np.array([float(note[0]) for note in notes])
    if len(X) <= 2:
        if max(X) - min(X) < 0.5:
            return "level"
        return "rising" if X[0] < X[1] else "falling"
    if max(X) - min(X) < 1.5:
        return "level"
    durations = [float(note[1]) for note in notes]
    Y = np.array([sum(durations[:i]) for i in range(len(durations))])
    Para = slovePara(X, Y)
    a, b, c = Para[0]
    symmetry = - b / (2 * a)
    if symmetry < X[0]:
        return "rising" if a > 0.0 else "falling"
    elif symmetry > X[-1]:
        return "falling" if a > 0.0 else "rising"
    else:
        return "falling rising" if a > 0.0 else "rising falling"


class Eval:
    tone_tone_direction_L3 = [[{0, 1, -1}, {0, -1}, {-1}, {1}, {0, -1}],
                           [{0, 1}, {0, 1}, {-1}, {1}, {0, -1}],
                           [{1}, {1}, {-1}, {1}, {1}],
                           [{1}, {1}, {-1}, {-1}, {-1}],
                           [{1}, {1}, {0, -1}, {1}, {0, 1}]]  # [wi-1][wi]
    tone_tone_direction_L5 = [[{0, 1, -1}, {0, -1}, {-1, -2}, {1}, {0, -1}],
                       [{0, 1}, {0, 1}, {-1, -2}, {1}, {0, -1}],
                       [{1, 2}, {1, 2}, {-1, -2}, {2}, {1, 2}],
                        [{2}, {2}, {-1, -2}, {-1, -2}, {-1, -2}],
                        [{0, 1}, {0, 1}, {-1, -2}, {1, 2}, {0, 1, -1}]]
    tone_shape_direction = [{"level"},
                            {"rising", "level"},
                 {"falling rising", "rising", "level"},
                 {"falling", "level"},
                 {"level"}]
    tone_to_pitch = [4, 2, 1, 5, 3]
    # stop_strings = ["[punc]", "[SEP]", "_"]
    stop_strings = ["[punc]", "[SEP]"]

    # PITCH PARAMS
    PITCH_CONTOUR_WEIGHT0 = 1  # contour趋势一致
    PITCH_CONTOUR_WEIGHT1 = 0  # contour趋势略有不同
    PITCH_CONTOUR_WEIGHT2 = 0  # contour趋势相反
    PITCH_SHAPE_WEIGHT0 = 1  # shape趋势一致
    PITCH_SHAPE_WEIGHT1 = 0  # shape趋势略有不同
    PITCH_SHAPE_WEIGHT2 = 0  # shape趋势相反
    
    # RHYTHM PARAMS
    SEP_WEIGHT = 0  # rest 为 sep
    PUNC_WEIGHT = 0  # rest 为 punc
    SPLIT_WEIGHT = 0  # rest 为 word split
    MISS_REST_WEIGHT = -1  # 出现rest但是没有出现对应的sep or punc带来的惩罚
    MIN_DURATION = 0.25  # 单个字可以接受的最小duration
    MIN_DURATION_WEIGHT = -5  # 小于min duration之后的penalty
    
    
    def __init__(self, notes, lyrics, rests, flag="L5"):
        self.flag = flag
        if flag == "L3":
            self.tone_tone_direction = self.tone_tone_direction_L3
        else:
            self.tone_tone_direction = self.tone_tone_direction_L5
        self.notes = [[n.split(":") for n in ng.split(" ")] for ng in notes.split("|")]  # note groups的形式
        self.lyrics = lyrics
        for i in self.stop_strings:
            self.lyrics = self.lyrics.replace(i, "")
        self.rests = [n.split(":") for n in rests.split(" ")]
        self.seps = self._get_pos(lyrics, "[SEP]")
        self.puncs = self._get_pos(lyrics, "[punc]")
        self.length = min(len(self.notes), len(self.lyrics))
        self.length_diff = len(self.notes) - len(self.lyrics)  # note的长度-lyrics的长度，>0表示note更长，反之lyrics更长
        self.note_directions = self._get_notes_directions()
        self.tones = self._get_tones()
        self.durs = self._get_durs()
        self.word_splits, self.words = self._get_splits()
    
    def _get_pos(self, lyrics, string):
        # pos是指该string在有效lyrics中第几个字之后
        txt = lyrics
        for i in self.stop_strings:  # 排除掉无关字符串的影响
            if i == string:
                continue
            txt = txt.replace(i, "")
        cur = 0
        pos = []
        for i in txt.split(string)[:-1]:
            cur += len(i)
            pos.append(cur)
        return pos
    
    def _get_tones(self):
        pinyin = lazy_pinyin(self.lyrics, style=Style.TONE3)
        tones = []  # 0 一声，1 二声，2 三声，3 四声，4 轻声
        for p in pinyin:
            if "1" <= p[-1] <= "4":
                tones.append(int(p[-1])-1)
            else:
                tones.append(4)
        return tones

    def _get_durs(self):
        durs = []
        for ng in self.notes:
            durs.append(sum([float(n[1]) for n in ng]))
        return durs
    
    def _get_splits(self):
        words = jieba.lcut(self.lyrics)
        last = 0
        idx = 0
        splits = []
        for word in words:
            cur = last + len(word)
            last = cur
            splits.append(cur)
        return splits, words
    
    def _get_notes_directions(self):
        directions = []
        last = float(self.notes[0][0][0])
        for i in range(1, self.length):
            cur = float(self.notes[i][0][0])
            if abs(cur-last) < 1:
                directions.append(0)
            elif cur > last:  # up
                if self.flag == "L5" and cur-last > 2:
                    directions.append(2)
                else:
                    directions.append(1)
            else:
                if self.flag == "L5" and last-cur > 2:
                    directions.append(-2)
                else:
                    directions.append(-1)
            last = cur
        return directions
    
    def _pitch_score_L5(self, delta_pitch, last, cur):
        sign = 1 if delta_pitch > 0 else -1
        if abs(delta_pitch) < 0.5:
            delta_level = 0
        elif abs(delta_pitch) <= 2.0:
            delta_level = 1
        else:
            delta_level = 2
        delta_level *= sign
        s = [abs(delta_level-x) for x in self.tone_tone_direction[last][cur]]
        return 1 - min(s)/4
    
    def _pitch_score_L3(self, delta_pitch, last, cur):
        assert NotImplementedError
    
    def pitch_contour_score_soft(self):
        # 未经过normalize的
        score = 0
        for i in range(self.length-1):
            delta_pitch = float(self.notes[i+1][0][0]) - float(self.notes[i][0][0])
            if self.flag == "L5":
                score += self._pitch_score_L5(delta_pitch, self.tones[i], self.tones[i+1])
            else:
                score += self._pitch_score_L3(delta_pitch, self.tones[i], self.tones[i+1])
        return score

    def pitch_contour_score_hard(self):
        # 未经过normalize的
        score = 0
        for i in range(self.length-1):
            last = self.tones[i]
            cur = self.tones[i+1]
            lyric_direction = self.tone_tone_direction[last][cur]
            if self.note_directions[i] in lyric_direction:
                score += self.PITCH_CONTOUR_WEIGHT0
            elif min(lyric_direction) - 1 <= self.note_directions[i] <= max(lyric_direction) + 1:
                score += self.PITCH_CONTOUR_WEIGHT1
            else:
                score += self.PITCH_CONTOUR_WEIGHT2
        return score
            
    
    def pitch_shape_score(self):
        score = 0
        ng_num = 0
        for idx, note_group in enumerate(self.notes):
            if len(note_group) <= 1:
                continue
            if len(self.tones) <= idx:
                break
            note_shape = notes2shape(note_group)
            tone_shape = self.tone_shape_direction[self.tones[idx]]
            if note_shape == "level":
                score += self.PITCH_SHAPE_WEIGHT0
            elif note_shape in tone_shape:
                score += self.PITCH_SHAPE_WEIGHT0
            else:
                score += self.PITCH_SHAPE_WEIGHT2
            ng_num += 1
        return score / ng_num if ng_num > 0 else 0
    
    def rhythm_char_score(self):
        # 单个音节不应该太短
        score = 0
        for idx, dur in enumerate(self.durs):
            if dur < self.MIN_DURATION:
                score += self.MIN_DURATION_WEIGHT * (self.MIN_DURATION - dur)
        return score
    
    def rhythm_word_score(self):
        # 分词，算duration分配
        score = 0
        start = 0
        for word in self.words:
            end = start + len(word)
            if start >= len(self.durs):
                break
            durs = self.durs[start:end]
            min_dur = min(durs)
            ratio = 0
            for d in durs:
                ratio += d / min_dur
            score += ratio / len(durs)
            start = end
        return score
    
    def rhythm_sent_score(self):
        score = 0
        if len(self.rests) == 0:
            return score
        missed_durations = 0
        rest_sent_nums = 0
        rest_nums = 0
        rest_miss_nums = 0
        total_duration = 0
        for i in self.rests:
            pos, duration = int(i[0]), float(i[1])
            if pos == -1 or pos == 0 or pos == len(self.notes) - 1:
                continue
            if pos in self.puncs:
                score += self.PUNC_WEIGHT
                total_duration += duration
                rest_nums += 1
            elif pos in self.seps:
                score += self.SEP_WEIGHT
                total_duration += duration
                rest_nums += 1
            elif pos in self.word_splits:
                score += self.SPLIT_WEIGHT
                total_duration += duration
                rest_nums += 1
            else:
                score += self.MISS_REST_WEIGHT
                missed_durations += duration
                rest_miss_nums += 1
            rest_sent_nums = 1
        return score, missed_durations, rest_sent_nums, rest_nums, rest_miss_nums, total_duration
    
    def get_align_scores(self, contour_flag="soft"):
        scores = {}
        if contour_flag == "soft":
            scores['pitch_contour'] = self.pitch_contour_score_soft() / self.length # 每个字都要算
        else:
            scores['pitch_contour'] = self.pitch_contour_score_hard() / self.length # 每个字都要算
        scores['pitch_shape'] = self.pitch_shape_score()
        scores['rhythm_sent'], scores['avg_miss_dur'], scores['rest_sent_nums'], scores['rest_nums'], scores['rest_miss_nums'], scores['avg_rest_dur'] = self.rhythm_sent_score()
        scores['rhythm_word'] = self.rhythm_word_score() / len(self.words) if len(self.words) > 0 else 0 
        scores['rhythm_char'] = self.rhythm_char_score() / self.length
        return scores
    
    def get_len_scores(self):
        return {"length_diff": self.length_diff, "length": len(self.lyrics)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('root_dir', type=str)
    parser.add_argument('--notes_dir', type=str, default=None, help="notes file")
    parser.add_argument('--split', type=str, default=None, help="notes file")
    parser.add_argument('--note', type=str, default="valid.notes.pitches_dur", help="notes file")
    parser.add_argument('--rest', type=str, default="valid.notes.rests_dur", help="rests file")
    parser.add_argument('--L', type=str, default="L5", help="pitch degree")
    parser.add_argument('--similarity', action="store_true", help="calculate the similarity")
    parser.add_argument('--contour', type=str, default="soft", help="hard or soft contour constraint")

    args = parser.parse_args()
    root_dir = args.root_dir
    if args.notes_dir is not None:
        assert args.split is not None 
        notes_file = os.path.join(args.notes_dir, f"{args.split}.notes.group.pitches_dur")
        rests_file = os.path.join(args.notes_dir, f"{args.split}.notes.group.rests_dur")
    else:
        notes_file = args.note
        rests_file = args.rest
    flag = args.L
    similairty_flag = args.similarity

    exp_dirs = glob.glob(f"{root_dir}/*/")
    print("Iterating", exp_dirs)

    params = []
    df_datas = {"exp_dir": [], "split": []}
    for exp_dir in exp_dirs:
        exp_dir = exp_dir.split("/")[-2]
        tmp = exp_dir.split("_")
        split = tmp[0]
        if args.split is not None:
            assert split == args.split
        lyric_file = os.path.join(root_dir, exp_dir, f"zh.{split}.hyp.org")
        ref_file = os.path.join(root_dir, exp_dir, f"zh.{split}.ref")
        src_file = os.path.join(root_dir, exp_dir, f"en.{split}.src")
        tgt_len_file = os.path.join(root_dir, exp_dir, f"{split}_tgt_len")
        if not os.path.exists(lyric_file):
            continue
        df_datas['exp_dir'].append(exp_dir)
        df_datas['split'].append(split)
        for pair in tmp[1:]:
            param, value = pair.split("-")
            if param not in df_datas.keys():
                df_datas[param] = []
            df_datas[param].append(value)
        
        lyrics_fn = open(lyric_file).readlines()
        ref_fn = open(ref_file).readlines()
        src_fn = open(src_file).readlines()
        notes_fn = open(notes_file).readlines()
        rests_fn = open(rests_file).readlines()
        tgt_len_fn = open(tgt_len_file).readlines()
        assert len(lyrics_fn) == len(notes_fn) == len(rests_fn) == len(tgt_len_fn)
        
        align_scores = {"pitch_contour": 0, "pitch_shape": 0, "rest_nums": 0, "rest_miss_nums": 0, "avg_miss_dur": 0, "avg_rest_dur": 0, "rhythm_word": 0, "rhythm_char": 0, "rest_sent_nums": 0}
        len_scores = []
        lyrics_invalid = []
        other_invalid = []
        unique_lyrics = []
        ref_lyrics = []
        src_lyrics = []
        punc_numbers = 0
        sep_numbers = 0
        ref_punc_numbers = 0
        ref_sep_numbers = 0
        
        for i in range(len(lyrics_fn)):
            note = notes_fn[i].replace("\n", "")
            lyric = preprocess_lyrics(lyrics_fn[i].replace("\n", ""))
            ref_line = preprocess_lyrics(ref_fn[i].replace("\n", ""))
            src_line = "] ".join(src_fn[i].replace("\n", "").split("] ")[2:])
            rest = rests_fn[i].replace("\n", "")
            tgt_len = int(tgt_len_fn[i].replace("\n", ""))
            if lyric is None or ref_line is None:
                lyrics_invalid.append(i)
                print("lyrics invalid", i, lyrics_fn[i].replace("\n", ""))
                continue
            try:
                e = Eval(note, lyric, rest, flag)
                align_score = e.get_align_scores(args.contour)
                len_score = e.get_len_scores()
                align_scores['pitch_contour'] += align_score['pitch_contour']
                align_scores['pitch_shape'] += align_score['pitch_shape']
#                 align_scores['miss_rest_num'] -= align_score['rhythm_sent']
                align_scores['rhythm_word'] += align_score['rhythm_word']
                align_scores['rhythm_char'] += align_score['rhythm_char']
                align_scores['rest_sent_nums'] += align_score['rest_sent_nums'] 
                align_scores['rest_nums'] += align_score['rest_nums'] 
                align_scores['rest_miss_nums'] += align_score['rest_miss_nums'] 
                align_scores['avg_rest_dur'] += align_score['avg_rest_dur'] 
                align_scores['avg_miss_dur'] += align_score['avg_miss_dur']
                len_score['length_diff'] = (tgt_len - len_score['length']) / tgt_len if tgt_len > 0 else 0
                len_scores.append(len_score)
                
                unique_lyrics.append(e.lyrics)
                ref_lyrics.append(ref_line.replace("[SEP]", "").replace("[punc]", ""))
                src_lyrics.append(src_line.replace("[SEP]", "").replace("[punc]", "").replace("  ", " "))
                punc_numbers += len(e.puncs)
                sep_numbers += len(e.seps)
                ref_punc_numbers += len(ref_line.split("[punc]")) - 1
                ref_sep_numbers += len(ref_line.split("[SEP]")) - 1
            except Exception as exc:
                print("[Error]", i, exc)
                other_invalid.append(i)
            
        valid_len = len(lyrics_fn) - len(other_invalid) - len(lyrics_invalid)
        print("================================ RUNNING STATISTICS ================================")
        print(f"Exp dir #{exp_dir}")
        print(f"Total #{len(lyrics_fn)}")
        print(f"Success #{valid_len}")
        print(f"Failed #{len(lyrics_invalid)} (invalid lyrics)")
        print(f"Failed #{len(other_invalid)} (other exception)")
        if "total_len" not in df_datas.keys():
            df_datas['total_len'] = []
        if "valid_len" not in df_datas.keys():
            df_datas['valid_len'] = []
        df_datas['total_len'].append(len(lyrics_fn))
        df_datas['valid_len'].append(valid_len)

        print("======================== ALIGN SCORES =======================")
        for k in align_scores.keys():
            if k in ['rest_sent_nums']:
                print(f"{k}\t:", align_scores[k])
                continue
            if k in ['rest_miss_nums', 'rest_nums']:
                res = align_scores[k] / align_scores['rest_sent_nums']
            elif k == 'avg_miss_dur':
                res = align_scores[k] / align_scores['rest_miss_nums']
            elif k == 'avg_rest_dur':
                res = align_scores[k] / align_scores['rest_nums']
            else:
                res = align_scores[k] / valid_len
            print(f"{k}\t:", res)
            if k not in df_datas.keys():
                df_datas[k] = []
            df_datas[k].append(res)

        print("========================= LEN SCORES ========================")
        notes_shorter_num = 0
        lyrics_shorter_num = 0
        notes_shorter_ratio = 0
        lyrics_shorter_ratio = 0
        for len_score in len_scores:
            if len_score['length_diff'] > 0:
                lyrics_shorter_num += 1
#                 lyrics_shorter_ratio += len_score['length_diff'] / len_score['length']
                lyrics_shorter_ratio += len_score['length_diff']
            elif len_score['length_diff'] < 0:
                notes_shorter_num += 1
#                 notes_shorter_ratio -= len_score['length_diff'] / len_score['length']
                notes_shorter_ratio -= len_score['length_diff']
        
        if "shorter_num" not in df_datas.keys():
            df_datas['shorter_num'] = []
        if "shorter_ratio" not in df_datas.keys():
            df_datas['shorter_ratio'] = []
        if "longer_num" not in df_datas.keys():
            df_datas['longer_num'] = []
        if "longer_ratio" not in df_datas.keys():
            df_datas['longer_ratio'] = []
            
        df_datas["shorter_num"].append(lyrics_shorter_num)
        if lyrics_shorter_num > 0:
            df_datas["shorter_ratio"].append(lyrics_shorter_ratio / lyrics_shorter_num)
        else:
            df_datas["shorter_ratio"].append(0)
        df_datas["longer_num"].append(notes_shorter_num)
        if notes_shorter_num > 0:
            df_datas["longer_ratio"].append(notes_shorter_ratio / notes_shorter_num)
        else:
            df_datas["longer_ratio"].append(0)
       
        print(f"Lyrcis shorter: #{lyrics_shorter_num}")
        print(f"Avg ratio: {df_datas['shorter_ratio'][-1]}")
        print(f"Lyrics longer: #{notes_shorter_num}")
        print(f"Avg ratio: {df_datas['longer_ratio'][-1]}")
        
        print("========================= SEP & PUNCS ========================")
        print(f"Puncs: #{punc_numbers} (in hyp) : #{ref_punc_numbers} (in ref)")
        print(f"Seps: #{sep_numbers} (in hyp) : #{ref_sep_numbers} (in ref)")
        if "punc_numbers" not in df_datas.keys():
            df_datas['punc_numbers'] = []
        if "sep_numbers" not in df_datas.keys():
            df_datas['sep_numbers'] = []
        if "ref_punc_numbers" not in df_datas.keys():
            df_datas['ref_punc_numbers'] = []
        if "ref_sep_numbers" not in df_datas.keys():
            df_datas['ref_sep_numbers'] = []
            
        df_datas["sep_numbers"].append(sep_numbers)
        df_datas["punc_numbers"].append(punc_numbers)
        df_datas["ref_sep_numbers"].append(ref_sep_numbers)
        df_datas["ref_punc_numbers"].append(ref_punc_numbers)
        
        print("========================= BLEU SCORES ========================")
        with open(os.path.join(root_dir, exp_dir, "ref"), "w") as f:
            f.write("\n".join(ref_lyrics))
        with open(os.path.join(root_dir, exp_dir, "hyp"), "w") as f:
            f.write("\n".join(unique_lyrics))
        cmd = f"python -m sacrebleu --force -lc -l zh-zh {root_dir}/{exp_dir}/ref < {root_dir}/{exp_dir}/hyp"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        result = p.communicate()[0].decode("utf-8")
        print(result)
        x = result.split("=")[1].split()[0]
        if "BLEU" not in df_datas.keys():
            df_datas['BLEU'] = []
        df_datas['BLEU'].append(float(x))
        
        if similairty_flag:
            print("====================== SENTENCE SIMILARITY =====================")
            with open(os.path.join(root_dir, exp_dir, "src"), "w") as f:
                f.write("\n".join(src_lyrics))
            cmd = f"bert-score --lang other -r {root_dir}/{exp_dir}/ref -c {root_dir}/{exp_dir}/hyp"
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            result = p.communicate()[0].decode("utf-8")
            print("ref & hyp", result)
            x = result.replace("\n", "").split(": ")[-1]
            if "sentence_similarity" not in df_datas.keys():
                df_datas['sentence_similarity'] = []
            df_datas['sentence_similarity'].append(float(x))

            cmd = f"bert-score --lang other -r {root_dir}/{exp_dir}/src -c {root_dir}/{exp_dir}/hyp"
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
            result = p.communicate()[0].decode("utf-8")
            print("src & hyp", result)
            x = result.replace("\n", "").split(": ")[-1]
            if "sentence_similarity_src" not in df_datas.keys():
                df_datas['sentence_similarity_src'] = []
            df_datas['sentence_similarity_src'].append(float(x))
    
    df = pd.DataFrame(data=df_datas)
    df.to_csv(os.path.join(root_dir, f"results.csv"), index=False)
