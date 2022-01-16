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
import sacrebleu
from scipy.optimize import leastsq


def preprocess_lyrics(lyrics):
    num_mappings = "零一二三四五六七八九"
    if bool(re.search(r"[a-zA-Z]", lyrics.replace("[punc]", "").replace("[sep]", ""))):  # 含有英文
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
    elif symmetry > x[-1]:
        return "falling" if a > 0.0 else "rising"
    else:
        return "falling rising" if a > 0.0 else "rising falling"
    
class Eval:
    # TODO：这两个directions可能要改
    tone_tone_direction = [[{0, 1, -1}, {0, -1}, {-1}, {1}, {0, -1}],
                           [{0, 1}, {0, 1}, {-1}, {1}, {0, -1}],
                           [{1}, {1}, {-1}, {1}, {1}],
                           [{1}, {1}, {-1}, {-1}, {-1}],
                           [{1}, {1}, {0, -1}, {1}, {0, 1}]]  # [wi-1][wi]
    tone_tone_direction_L5 = [[{0, 1, -1}, {0, -1}, {-1, -2}, {1}, {0, -1}],
                       [{0, 1}, {0, 1}, {-1, -2}, {1}, {0, -1}],
                       [{1, 2}, {1, 2}, {-1, -2}, {2}, {1, 2}],
                        [{2}, {2}, {-1, -2}, {-1, -2}, {-1, -2}],
                        [{0, 1}, {0, 1}, {-1, -2}, {1, 2}, {0, 1, -1}]]
#     tone_shape_direction = [{0, 1, -1}, {1}, {0, -1, 1}, {-1}, {0}]
    tone_shape_direction = [{"level"},
                            {"rising", "level"},
                 {"falling rising", "rising", "level"},
                 {"falling", "level"},
                 {"level"}]
    tone_to_pitch = [4, 2, 1, 5, 3]
    stop_strings = ["[punc]", "[sep]", "_"]
    
    # PITCH PARAMS
    PITCH_CONTOUR_WEIGHT0 = 2  # contour趋势一致
    PITCH_CONTOUR_WEIGHT1 = 0  # contour趋势略有不同
    PITCH_CONTOUR_WEIGHT2 = -1  # contour趋势相反
    PITCH_SHAPE_WEIGHT0 = 2  # shape趋势一致
    PITCH_SHAPE_WEIGHT1 = 0  # shape趋势略有不同
    PITCH_SHAPE_WEIGHT2 = -1  # shape趋势相反
    
    # RHYTHM PARAMS
    SEP_WEIGHT = 0  # rest 为 sep
    PUNC_WEIGHT = 0  # rest 为 punc
    SPLIT_WEIGHT = 0  # rest 为 word split
    MISS_REST_WEIGHT = -1  # 出现rest但是没有出现对应的sep or punc带来的惩罚
    MIN_DURATION = 0.25  # 单个字可以接受的最小duration
    MIN_DURATION_WEIGHT = -5  # 小于min duration之后的penalty
    
    def __init__(self, notes, lyrics, rests):
        self.notes = [n.split(":") for n in notes.split(" ")]
        self.lyrics = lyrics
        for i in self.stop_strings:
            self.lyrics = self.lyrics.replace(i, "")
        self.rests = [n.split(":") for n in rests.split(" ")]
        self.seps = self._get_pos(lyrics, "[sep]")
        self.puncs = self._get_pos(lyrics, "[punc]")
        self.junctions = self._get_pos(lyrics, "_")
        self.lyrics_unique = self._get_unique_lyrics()
        self.length = min(len(self.notes), len(self.lyrics_unique))
        self.length_diff = len(self.notes) - len(self.lyrics_unique)  # note的长度-lyrics的长度，>0表示note更长，反之lyrics更长
        self.tones = self._get_tones()
        self.durs = self._get_durs()
        self.lyrics_directions = self._get_lyrics_directions()
        self.note_directions = self._get_notes_directions()
        self.word_splits, self.words = self._get_splits()
        
    def _get_pos(self, lyrics, string):
        txt = lyrics
        for i in self.stop_strings:
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
    
    def _get_unique_lyrics(self):
        lyrics = self.lyrics
        for junction in self.junctions[::-1]:
            lyrics = lyrics[:junction] + lyrics[junction+1:]
        return lyrics
    
    def _get_lyrics_directions(self):
        directions = []  # len(directions) = self.length - 1
        last = self.tones[0]
        for i in range(1, self.length):
            cur = self.tones[i]
            directions.append(self.tone_tone_direction[last][cur])
            last = cur
        return directions
    
    def _get_notes_directions(self):
        directions = []
        last = float(self.notes[0][0])
        for i in range(1, self.length):
            cur = float(self.notes[i][0])
            if cur == last:
                directions.append(0)
            elif cur > last:  # up
                directions.append(1)
            else:
                directions.append(-1)
            last = cur
        return directions
    
    def _get_durs(self):
        durs = []
        for note in self.notes:
            durs.append(float(note[1]))
        for junction in self.junctions[::-1]:
            durs[junction-1] += durs[junction]
            durs = durs[:junction] + durs[junction+1:]
        return durs
    
    def _get_splits(self):
        words = jieba.lcut(self.lyrics_unique)
        last = 0
        idx = 0
        splits = []
        for word in words:
            cur = last + len(word)
            if idx < len(self.junctions) and last < self.junctions[idx] <= cur:
                cur += 1
                idx += 1
            last = cur
            splits.append(cur)
        return splits, words
    
    def pitch_contour_score(self):
        score = 0
        for i in range(self.length-1):
            if i in self.junctions:  # 表明是内部的note，直接不用管
                continue  # TODO: 这里是直接当做0还是应该给加分？
            if self.note_directions[i] in self.lyrics_directions[i]:
                score += self.PITCH_CONTOUR_WEIGHT0
            elif min(self.lyrics_directions[i]) - 1 <= self.note_directions[i] <= max(self.lyrics_directions[i]) + 1:
                score += self.PITCH_CONTOUR_WEIGHT1
            else:
                score += self.PITCH_CONTOUR_WEIGHT2
        return score
    
    def pitch_shape_score(self):
        score = 0
        # 先按照junction group出一些notes
        last = -2
        notes = []
        tones = []
        for junction in self.junctions:
            assert self.lyrics[junction-1] == self.lyrics[junction]  # 保证是同一个字
            if junction - last == 1:
                notes[-1].append(self.notes[junction])
            else:
                notes.append(self.notes[junction-1:junction+1])
                tones.append(self.tones[junction])
        # 然后分别由notes判断走向
        for idx, note_group in enumerate(notes):
            note_shape = notes2shape(note_group)
            tone_shape = self.tone_shape_direction[tones[idx]]
            # 赋分
            if note_shape == "level":
                pass
            elif note_shape in tone_shape:
                score += self.PITCH_SHAPE_WEIGHT0
            else:
                score += self.PITCH_SHAPE_WEIGHT2
            
#         for junction in self.junctions:
#             assert self.lyrics[junction-1] == self.lyrics[junction]  # 保证是同一个字
#             note_shape = self.note_directions[junction-1]
#             # TODO: 如果一直level就算对，那这样的话应该直接当做0还是应该给加分？
#             if note_shape == 0:
#                 continue
#             tone = self.tones[junction]
#             tone_shape = self.tone_shape_direction[tone]
#             if note_shape in tone_shape:
#                 score += self.PITCH_SHAPE_WEIGHT0
#             elif min(tone_shape) - 1 < note_shape < max(tone_shape) + 1:
#                 score += self.PITCH_SHAPE_WEIGHT1
#             else:
#                 score += self.PITCH_SHAPE_WEIGHT2
        return score
    
    def rhythm_sent_score(self):
        score = 0
        if len(self.rests) == 0:
            return score
        missed_durations = 0
        for i in self.rests:
            pos, duration = int(i[0]), float(i[1])
            if pos == -1:
                continue
            if pos in self.puncs:
                score += self.PUNC_WEIGHT
            elif pos in self.seps:
                score += self.SEP_WEIGHT
            elif pos in self.word_splits:
                score += self.SPLIT_WEIGHT
            else:
#                 score += self.MISS_REST_WEIGHT * duration
                score += self.MISS_REST_WEIGHT
                missed_durations += duration
        return score, missed_durations / (score / self.MISS_REST_WEIGHT) if missed_durations != 0 else 0
    
    def rhythm_char_score(self):
        # for 每个字，算duration （注意考虑一字多音）
        score = 0
        for idx, dur in enumerate(self.durs):
            if dur < self.MIN_DURATION and idx not in self.junctions:
#                 score += self.MIN_DURATION_WEIGHT * (self.MIN_DURATION - dur)
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
#             if max(durs) != durs[-1]:
#                 score -= 
            min_dur = min(durs)
            ratio = 0
            for d in durs:
                ratio += d / min_dur
            score += ratio / len(durs)
            start = end
        return score
    
    def get_align_scores(self):
        scores = {}
        scores['pitch_contour'] = self.pitch_contour_score() / self.length
        scores['pitch_shape'] = self.pitch_shape_score() / len(self.junctions) if len(self.junctions) > 0 else 0 
        scores['rhythm_sent'], scores['avg_miss_dur'] = self.rhythm_sent_score()
        scores['rhythm_word'] = self.rhythm_word_score() / len(self.words) if len(self.words) > 0 else 0 
        scores['rhythm_char'] = self.rhythm_char_score() / self.length
        
        return scores
    
    def get_len_scores(self):
        return {"length_diff": self.length_diff, "length": self.length}
    

# Usage: python eval.py <root_dir> <notes_file> <rests_file>
if __name__ == "__main__":
    root_dir = sys.argv[1]
    notes_file = sys.argv[2]
    rests_file = sys.argv[3]

    exp_dirs = glob.glob(f"{root_dir}/*/")
    print("Iterating", exp_dirs)

    df_datas = {"exp_dir": [], "split": []}
    for exp_dir in exp_dirs:
        exp_dir = exp_dir.split("/")[-2]
        df_datas['exp_dir'].append(exp_dir)
        tmp = exp_dir.split("_")
        split = tmp[0]
        df_datas['split'].append(split)
        for pair in tmp[1:]:
            param, value = pair.split("-")
            if param not in df_datas.keys():
                df_datas[param] = []
            df_datas[param].append(value)
        lyric_file = os.path.join(exp_dir, f"zh.{split}.hyp")
        ref_file = os.path.join(exp_dir, f"zh.{split}.ref")
        tgt_len_file = os.path.join(exp_dir, f"{split}_tgt_len")
        
        lyrics_fn = open(lyric_file).readlines()
        ref_fn = open(ref_file).readlines()
        notes_fn = open(notes_file).readlines()
        rests_fn = open(rests_file).readlines()
        tgt_len_fn = open(tgt_len_file).readlines()
        assert len(lyrics_fn) == len(notes_fn) == len(rests_fn) == len(tgt_len_fn)
        
        align_scores = {"pitch_contour": 0, "pitch_shape": 0, "miss_rest_num": 0, "total_rest_num": 0, "avg_miss_dur": 0, "rhythm_word": 0, "rhythm_char": 0}
        len_scores = []
        lyrics_invalid = []
        other_invalid = []
        unique_lyrics = []
        ref_lyrics = []
        punc_numbers = 0
        sep_numbers = 0
        ref_punc_numbers = 0
        ref_sep_numbers = 0
        
        for i in range(len(lyrics_fn)):
            note = notes_fn[i].replace("\n", "")
            lyric = preprocess_lyrics(lyrics_fn[i].replace("\n", ""))
            ref_line = preprocess_lyrics(ref_fn[i].replace("\n", ""))
            rest = rests_fn[i].replace("\n", "")
            tgt_len = int(tgt_len_fn[i].replace("\n", ""))
            if lyric is None or ref_line is None:
                lyrics_invalid.append(i)
                print("lyrics invalid", i, lyrics_fn[i].replace("\n", ""))
                continue
            try:
                e = Eval(note, lyric, rest)
                align_score = e.get_align_scores()
                len_score = e.get_len_scores()
                align_scores['pitch_contour'] += align_score['pitch_contour']
                align_scores['pitch_shape'] += align_score['pitch_shape']
                align_scores['miss_rest_num'] -= align_score['rhythm_sent']
                align_scores['total_rest_num'] += len(e.rests)
                align_scores['avg_miss_dur'] += align_score['avg_miss_dur']
                align_scores['rhythm_word'] += align_score['rhythm_word']
                align_scores['rhythm_char'] += align_score['rhythm_char']
                len_scores.append(len_score)
                
                unique_lyrics.append(e.lyrics_unique)
                ref_lyrics.append(ref_line.replace("[sep]", "").replace("[punc]", ""))
                punc_numbers += len(e.puncs)
                sep_numbers += len(e.seps)
                ref_punc_numbers += len(ref_line.split("[punc]")) - 1
                ref_sep_numbers += len(ref_line.split("[sep]")) - 1
            except Exception as e:
                print("[Error]", i, e)
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
            print(f"{k}\t:", align_scores[k] / valid_len)
            if k not in df_datas.keys():
                df_datas[k] = []
            df_datas[k].append(align_scores[k] / valid_len)

        print("========================= LEN SCORES ========================")
        notes_shorter_num = 0
        lyrics_shorter_num = 0
        notes_shorter_ratio = 0
        lyrics_shorter_ratio = 0
        for len_score in len_scores:
            if len_score['length_diff'] > 0:
                lyrics_shorter_num += 1
#                 lyrics_shorter_ratio += len_score['length_diff'] / len_score['length']
                lyrics_shorter_ratio += len_score['length_diff'] / tgt_len
            elif len_score['length_diff'] < 0:
                notes_shorter_num += 1
#                 notes_shorter_ratio -= len_score['length_diff'] / len_score['length']
                notes_shorter_ratio -= len_score['length_diff'] / tgt_len
        
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
        with open(os.path.join(exp_dir, "ref"), "w") as f:
            f.write("\n".join(ref_lyrics))
        with open(os.path.join(exp_dir, "hyp"), "w") as f:
            f.write("\n".join(unique_lyrics))
        cmd = f"python -m sacrebleu --force -lc -l zh-zh {exp_dir}/ref < {exp_dir}/hyp"
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        result = p.communicate()[0].decode("utf-8")
        print(result)
        x = result.split("=")[1].split()[0]
        if "BLEU" not in df_datas.keys():
            df_datas['BLEU'] = []
        df_datas['BLEU'].append(float(x))
    
    
    df = pd.DataFrame(data=df_datas)
#     print(df_datas)
    df.to_csv(os.path.join(root_dir, f"results.csv"), index=False)
