# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
import regex
import fugashi
import unidic

from tqdm import tqdm


df = df = pd.read_csv('../data/jyouyou_kanji_with_grade.csv', header=0)
df.columns = ['kanji', 'grade']

grade_1 = df[df['grade'] == 1]['kanji'].tolist()
grade_2 = df[df['grade'] == 2]['kanji'].tolist()
grade_3 = df[df['grade'] == 3]['kanji'].tolist()
grade_4 = df[df['grade'] == 4]['kanji'].tolist()
grade_5 = df[df['grade'] == 5]['kanji'].tolist()
grade_6 = df[df['grade'] == 6]['kanji'].tolist()
grade_7 = df[df['grade'] == 7]['kanji'].tolist()

grade_1_re = '[' + ''.join(grade_1) + ']+'
grade_2_re = '[' + ''.join(grade_1 + grade_2) + ']+'
grade_3_re = '[' + ''.join(grade_1 + grade_2 + grade_3) + ']+'
grade_4_re = '[' + ''.join(grade_1 + grade_2 + grade_3 + grade_4) + ']+'
grade_5_re = '[' + ''.join(grade_1 + grade_2 + grade_3 + grade_4 + grade_5) + ']+'
grade_6_re = '[' + ''.join(grade_1 + grade_2 + grade_3 + grade_4 + grade_5 + grade_6) + ']+'
grade_7_re = '[' + ''.join(grade_1 + grade_2 + grade_3 + grade_4 + grade_5 + grade_6 + grade_7) + ']+'

grade_1_only_re = '[' + ''.join(grade_1) + ']+'
grade_2_only_re = '[' + ''.join(grade_2) + ']+'
grade_3_only_re = '[' + ''.join(grade_3) + ']+'
grade_4_only_re = '[' + ''.join(grade_4) + ']+'
grade_5_only_re = '[' + ''.join(grade_5) + ']+'
grade_6_only_re = '[' + ''.join(grade_6) + ']+'
grade_7_only_re = '[' + ''.join(grade_7) + ']+'

RE_GRADE_1 = regex.compile(grade_1_re)
RE_GRADE_2 = regex.compile(grade_2_re)
RE_GRADE_3 = regex.compile(grade_3_re)
RE_GRADE_4 = regex.compile(grade_4_re)
RE_GRADE_5 = regex.compile(grade_5_re)
RE_GRADE_6 = regex.compile(grade_6_re)
RE_GRADE_7 = regex.compile(grade_7_re)

RE_GRADE_1_ONLY = regex.compile(grade_1_only_re)
RE_GRADE_2_ONLY = regex.compile(grade_2_only_re)
RE_GRADE_3_ONLY = regex.compile(grade_3_only_re)
RE_GRADE_4_ONLY = regex.compile(grade_4_only_re)
RE_GRADE_5_ONLY = regex.compile(grade_5_only_re)
RE_GRADE_6_ONLY = regex.compile(grade_6_only_re)
RE_GRADE_7_ONLY = regex.compile(grade_7_only_re)

RE_GRADE_MAP = {
    1: RE_GRADE_1,
    2: RE_GRADE_2,
    3: RE_GRADE_3,
    4: RE_GRADE_4,
    5: RE_GRADE_5,
    6: RE_GRADE_6,
    7: RE_GRADE_7
}

RE_KANJI = regex.compile(r'\p{Script=Han}+')
RE_HIRAGANA = regex.compile(r'\p{Script=Hiragana}+')
RE_KATAKANA = regex.compile(r'\p{Script=Katakana}+')
RE_NUMERIC = regex.compile(r'\p{N}+')
RE_LATIN = regex.compile(r'\p{Script=Latin}+')
RE_PUNCTUATION = regex.compile(r'\p{P}+')
KANJI_RUN = regex.compile(r"\p{Han}+", flags=regex.V1)

# function to convert Katakana to Hiragana
def katakana_to_hiragana(text):
    return "".join([chr(ord(char) - 96) if "ァ" <= char <= "ヶ" else char for char in text])

def is_kanji(text):
    return RE_KANJI.fullmatch(text) is not None

def is_hiragana(text):
    return RE_HIRAGANA.fullmatch(text) is not None

def is_katakana(text):
    return RE_KATAKANA.fullmatch(text) is not None

def has_hiragana(text):
    return RE_HIRAGANA.search(text) is not None

def has_katakana(text):
    return RE_KATAKANA.search(text) is not None

def has_kanji(text):
    return RE_KANJI.search(text) is not None

def annotate(orig: str, hira: str, allowed: set[str] | None, 
             ruby_l="<rt>", ruby_r="</rt>",
             kanji_l="<ruby>", kanji_r="</ruby>"):
    pat_parts, tpl_parts = [], []
    last = 0
    cap_count = 0

    for km in KANJI_RUN.finditer(orig):
        lit = orig[last:km.start()]
        pat_parts.append(regex.escape(lit))
        tpl_parts.append(lit)

        run = km.group(0)

        # allowed=None => annotate all kanji runs
        should_annotate = True if allowed is None else any(ch not in allowed for ch in run)

        if should_annotate:
            pat_parts.append(r"(\p{Hiragana}+?)")
            tpl_parts.append(f"{kanji_l}{run}" + "{}")
            cap_count += 1
        else:
            pat_parts.append(r"(?:\p{Hiragana}+?)")
            tpl_parts.append(run)

        last = km.end()

    tail = orig[last:]
    pat_parts.append(regex.escape(tail))
    tpl_parts.append(tail)

    pattern = "".join(pat_parts)
    template = "".join(tpl_parts)

    m = regex.fullmatch(pattern, hira, flags=regex.V1)
    if not m or cap_count == 0:
        return orig

    return template.format(*[f"{ruby_l}{x}{ruby_r}{kanji_r}" for x in m.groups()])


def okurigana(orig: str, hira: str, df, level: int = 0, 
              ruby_l="<rt>", ruby_r="</rt>", kanji_l="<ruby>", kanji_r="</ruby>"):
    if level == 0:
        allowed = None  # sentinel: annotate all
    else:
        allowed = set(df.loc[df["grade"].between(1, level), "kanji"].astype(str))

    return annotate(orig, hira, allowed, ruby_l, ruby_r, kanji_l, kanji_r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input_file',
                        required=True,
                        help='input file')
    parser.add_argument('-a',
                        '--add_ruby',
                        action='store_true',
                        help='add ruby tags to kanji with okurigana')
    parser.add_argument('-l',
                        '--level',
                        default=0,
                        type=int,
                        help='Kanji level to transliterate (0-7), default=0')
    parser.add_argument('-s',
                        '--save_sample_html',
                        action='store_true',
                        help='save sample html file with ruby annotations')

    args = parser.parse_args()

    # Load the input file
    with open(args.input_file, 'r') as f:
        print('reading from', args.input_file)
        lines = [line.strip() for line in f.readlines()]

    tagger = fugashi.Tagger('-d "{}"'.format(unidic.DICDIR))

    # transliterate the lines
    only_kana = []

    total_lines = len(lines)
    print('transliterating {} lines'.format(total_lines))

    total_chars_processed = 0
    total_chars_hiragana = 0
    total_chars_katakana = 0
    total_chars_kanji = 0
    total_chars_numeric = 0
    total_chars_latin = 0
    total_chars_punctuation = 0
    total_chars_level_1_only = 0
    total_chars_level_2_only = 0
    total_chars_level_3_only = 0
    total_chars_level_4_only = 0
    total_chars_level_5_only = 0
    total_chars_level_6_only = 0
    total_chars_level_7_only = 0
    total_chars_level_1 = 0
    total_chars_level_2 = 0
    total_chars_level_3 = 0
    total_chars_level_4 = 0
    total_chars_level_5 = 0
    total_chars_level_6 = 0
    total_chars_level_7 = 0

    total_words = 0
    total_words_transliterated = 0
    total_words_failed_transliterated = 0
    
    for line in tqdm(lines):
        kana_line = []
        # count characters
        total_chars_processed += len(line)
        total_chars_hiragana += sum(len(m) for m in RE_HIRAGANA.findall(line))
        total_chars_katakana += sum(len(m) for m in RE_KATAKANA.findall(line))
        total_chars_kanji += sum(len(m) for m in RE_KANJI.findall(line))
        total_chars_numeric += sum(len(m) for m in RE_NUMERIC.findall(line))
        total_chars_latin += sum(len(m) for m in RE_LATIN.findall(line))
        total_chars_punctuation += sum(len(m) for m in RE_PUNCTUATION.findall(line))
        
        total_chars_level_1 += sum(len(m) for m in RE_GRADE_1.findall(line))
        total_chars_level_2 += sum(len(m) for m in RE_GRADE_2.findall(line))
        total_chars_level_3 += sum(len(m) for m in RE_GRADE_3.findall(line))
        total_chars_level_4 += sum(len(m) for m in RE_GRADE_4.findall(line))
        total_chars_level_5 += sum(len(m) for m in RE_GRADE_5.findall(line))
        total_chars_level_6 += sum(len(m) for m in RE_GRADE_6.findall(line))
        total_chars_level_7 += sum(len(m) for m in RE_GRADE_7.findall(line))

        total_chars_level_1_only += sum(len(m) for m in RE_GRADE_1_ONLY.findall(line))
        total_chars_level_2_only += sum(len(m) for m in RE_GRADE_2_ONLY.findall(line))
        total_chars_level_3_only += sum(len(m) for m in RE_GRADE_3_ONLY.findall(line))
        total_chars_level_4_only += sum(len(m) for m in RE_GRADE_4_ONLY.findall(line))
        total_chars_level_5_only += sum(len(m) for m in RE_GRADE_5_ONLY.findall(line))
        total_chars_level_6_only += sum(len(m) for m in RE_GRADE_6_ONLY.findall(line))
        total_chars_level_7_only += sum(len(m) for m in RE_GRADE_7_ONLY.findall(line))

        for word in tagger(line):
            kana = word.surface
            if has_kanji(word.surface):
                kana = word.feature.kana
                if not kana:
                    #print('Warning: No kana reading for word:', word.surface)
                    kana = word.surface
                    total_words_failed_transliterated += 1
                else:
                    if args.level > 0:
                        # check if all the letters in word.surface are in grade x
                        if RE_GRADE_MAP[args.level].fullmatch(word.surface) is None:
                            kana = katakana_to_hiragana(kana)
                            total_words_transliterated += 1
                            if args.add_ruby:
                                ruby = okurigana(word.surface, kana, df, level=args.level)
                                kana = ruby
                        else:
                            kana = word.surface
                    else:
                        kana = katakana_to_hiragana(kana)
                        total_words_transliterated += 1
                        if args.add_ruby:
                            ruby = okurigana(word.surface, kana, df, level=args.level)
                            kana = ruby
            kana_line.append(kana)
            total_words += 1
        only_kana.append(''.join(kana_line))

    print('total words processed:', total_words)
    print('total words transliterated:', total_words_transliterated)
    print('total words failed to transliterate:', total_words_failed_transliterated)

    print('total characters processed:', total_chars_processed)
    print('total hiragana characters:', total_chars_hiragana)
    print('total katakana characters:', total_chars_katakana)

    print('total kanji characters:', total_chars_kanji)
    print('total kanji characters with grade 1:', total_chars_level_1_only)
    print('total kanji characters with grade 2:', total_chars_level_2_only)
    print('total kanji characters with grade 3:', total_chars_level_3_only)
    print('total kanji characters with grade 4:', total_chars_level_4_only)
    print('total kanji characters with grade 5:', total_chars_level_5_only)
    print('total kanji characters with grade 6:', total_chars_level_6_only)
    print('total kanji characters with jyouyou:', total_chars_level_7_only)

    print('total kanji characters up to grade 1:', total_chars_level_1)
    print('total kanji characters up to grade 2:', total_chars_level_2)
    print('total kanji characters up to grade 3:', total_chars_level_3)
    print('total kanji characters up to grade 4:', total_chars_level_4)
    print('total kanji characters up to grade 5:', total_chars_level_5)
    print('total kanji characters up to grade 6:', total_chars_level_6)
    print('total kanji characters up to jyouyou:', total_chars_level_7)

    print('total numeric characters:', total_chars_numeric)
    print('total latin characters:', total_chars_latin)
    print('total punctuation characters:', total_chars_punctuation)

    # write to files
    print('writing to files')

    if args.input_file.endswith('.txt'):
        if args.input_file.endswith('.R.txt'):
            input_file = args.input_file[:-6]
        else:
            input_file = args.input_file[:-4]

    if args.add_ruby:
        extension = ".F"
        extension += str(args.level)
    else:
        extension = ".K"
        extension += str(args.level)

    output_file = input_file + extension  + '.txt'
    with open(output_file, 'w') as f:
        print('writing to', output_file)
        for line in only_kana:
            f.write(line + '\n')

    # save stats to json
    stats_dict = {
        'file_name': args.input_file,
        'level': args.level,
        'total_lines': total_lines,
        'total_words': total_words,
        'total_words_transliterated': total_words_transliterated,
        'total_words_failed_to_transliterate': total_words_failed_transliterated,
        'total_chars_processed': total_chars_processed,
        'total_chars_hiragana': total_chars_hiragana,
        'total_chars_katakana': total_chars_katakana,
        'total_chars_kanji': total_chars_kanji,
        'total_chars_level_1_only': total_chars_level_1_only,
        'total_chars_level_2_only': total_chars_level_2_only,
        'total_chars_level_3_only': total_chars_level_3_only,
        'total_chars_level_4_only': total_chars_level_4_only,
        'total_chars_level_5_only': total_chars_level_5_only,
        'total_chars_level_6_only': total_chars_level_6_only,
        'total_chars_level_7_only': total_chars_level_7_only,
        'total_chars_level_1': total_chars_level_1,
        'total_chars_level_2': total_chars_level_2,
        'total_chars_level_3': total_chars_level_3,
        'total_chars_level_4': total_chars_level_4,
        'total_chars_level_5': total_chars_level_5,
        'total_chars_level_6': total_chars_level_6,
        'total_chars_level_7': total_chars_level_7,
        'total_chars_numeric': total_chars_numeric,
        'total_chars_latin': total_chars_latin,
        'total_chars_punctuation': total_chars_punctuation
    }

    input_dir = os.path.dirname(args.input_file)
    output_file_name = os.path.basename(input_file + extension)
    output_file_name = "stats." + output_file_name + '.json'
    output_file = os.path.join(input_dir, output_file_name)

    with open(output_file, 'w') as f:
        print('writing to', output_file)
        json.dump(stats_dict, f, ensure_ascii=False, indent=4)

    if args.save_sample_html:
        sample_html_file = input_file + extension +  '.html'
        with open(sample_html_file, 'w') as f:
            print('writing sample html to', sample_html_file)
            f.write('<html><body>\n')
            for i, line in enumerate(only_kana[:1000]):
                f.write(f'<p><b>{i}:</b> {line}</p>\n')
            f.write('</body></html>\n')
