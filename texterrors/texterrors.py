#!/usr/bin/env python
import sys
from collections import defaultdict
import texterrors_align
import numpy as np
import plac
from loguru import logger
from termcolor import colored
import re
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import os


def convert_to_int(lst_a, lst_b, dct):
    def convert(lst, dct_syms):
        intlst = []
        for w in lst:
            if w not in dct:
                i = max(v for v in dct_syms.values() if isinstance(v, int)) + 1
                dct_syms[w] = i
                dct_syms[i] = w
            intlst.append(dct_syms[w])
        return intlst
    int_a = convert(lst_a, dct)
    int_b = convert(lst_b, dct)
    return int_a, int_b


def lev_distance(a, b):
    if isinstance(a, str):
        return texterrors_align.lev_distance_str(a, b)
    else:
        return texterrors_align.lev_distance(a, b)


def _align_texts(text_a_str, text_b_str, use_chardiff, debug, insert_tok):
    len_a = len(text_a_str)
    len_b = len(text_b_str)
    # doing dynamic time warp
    text_a_str = [insert_tok] + text_a_str
    text_b_str = [insert_tok] + text_b_str
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost(summed_cost, text_a_str, text_b_str, use_chardiff)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')
    best_path_lst = []
    texterrors_align.get_best_path(summed_cost, best_path_lst, text_a_str, text_b_str, use_chardiff)
    assert len(best_path_lst) % 2 == 0
    path = []
    for n in range(0, len(best_path_lst), 2):
        i = best_path_lst[n]
        j = best_path_lst[n + 1]
        path.append((i, j))

    # convert hook (up left or left up) transitions to diag, not important.
    # -1 because of padding tokens, i = 1 because first is given
    newpath = [path[0]]
    i = 1
    lasttpl = path[0]
    while i < len(path) - 1:
        tpl = path[i]
        nexttpl = path[i + 1]
        if (
            lasttpl[0] - 1 == nexttpl[0] and lasttpl[1] - 1 == nexttpl[1]
        ):  # minus because reversed
            pass
        else:
            newpath.append(tpl)
        i += 1
        lasttpl = tpl
    path = newpath

    aligned_a, aligned_b = [], []
    lasti, lastj = -1, -1
    for i, j in list(reversed(path)):
        # print(text_a[i], text_b[i], file=sys.stderr)
        if i != lasti:
            aligned_a.append(text_a_str[i])
        else:
            aligned_a.append(insert_tok)
        if j != lastj:
            aligned_b.append(text_b_str[j])
        else:
            aligned_b.append(insert_tok)
        lasti, lastj = i, j

    return aligned_a, aligned_b, cost


def _align_texts_ctm(text_a_str, text_b_str, times_a, times_b, use_chardiff, debug, insert_tok):
    len_a = len(text_a_str)
    len_b = len(text_b_str)
    # doing dynamic time warp
    text_a_str = [insert_tok] + text_a_str
    text_b_str = [insert_tok] + text_b_str
    # +1 because of padded start token
    summed_cost = np.zeros((len_a + 1, len_b + 1), dtype=np.float64, order="C")
    cost = texterrors_align.calc_sum_cost_ctm(summed_cost, text_a_str, text_b_str, times_a, times_b,
        use_chardiff)

    if debug:
        np.set_printoptions(linewidth=300)
        np.savetxt('summedcost', summed_cost, fmt='%.3f', delimiter='\t')
    best_path_lst = []
    texterrors_align.get_best_path_ctm(summed_cost, best_path_lst, text_a_str, text_b_str, times_a,
        times_b, use_chardiff)
    assert len(best_path_lst) % 2 == 0
    path = []
    for n in range(0, len(best_path_lst), 2):
        i = best_path_lst[n]
        j = best_path_lst[n + 1]
        path.append((i, j))

    # convert hook (up left or left up) transitions to diag, not important.
    # -1 because of padding tokens, i = 1 because first is given
    newpath = [path[0]]
    i = 1
    lasttpl = path[0]
    while i < len(path) - 1:
        tpl = path[i]
        nexttpl = path[i + 1]
        if (
            lasttpl[0] - 1 == nexttpl[0] and lasttpl[1] - 1 == nexttpl[1]
        ):  # minus because reversed
            pass
        else:
            newpath.append(tpl)
        i += 1
        lasttpl = tpl
    path = newpath

    aligned_a, aligned_b = [], []
    lasti, lastj = -1, -1
    for i, j in list(reversed(path)):
        # print(text_a[i], text_b[i], file=sys.stderr)
        if i != lasti:
            aligned_a.append(text_a_str[i])
        else:
            aligned_a.append(insert_tok)
        if j != lastj:
            aligned_b.append(text_b_str[j])
        else:
            aligned_b.append(insert_tok)
        lasti, lastj = i, j

    return aligned_a, aligned_b, cost


def align_texts(text_a, text_b, debug, insert_tok='<eps>', use_chardiff=True):

    assert isinstance(text_a, list) and isinstance(text_b, list), 'Input types should be a list!'
    assert isinstance(text_a[0], str)

    aligned_a, aligned_b, cost = _align_texts(text_a, text_b, use_chardiff,
                                              debug=debug, insert_tok=insert_tok)

    if debug:
        print(aligned_a)
        print(aligned_b)
    return aligned_a, aligned_b, cost


def get_overlap(refw, hypw):
    # 0 if match, -1 if hyp before, 1 if after
    if hypw[1] < refw[1]:
        neg_offset = refw[1] - hypw[1]
        if neg_offset < hypw[2] * 0.5:
            return 0
        else:
            return -1
    else:
        pos_offset = hypw[1] - refw[1]
        if pos_offset < hypw[2] * 0.5:
            return 0
        else:
            return 1


def get_oov_cer(ref_aligned, hyp_aligned, oov_set):
    assert len(ref_aligned) == len(hyp_aligned)
    oov_count_denom = 0
    oov_count_error = 0
    for i, ref_w in enumerate(ref_aligned):
        if ref_w in oov_set:
            oov_count_denom += len(ref_w)
            startidx = i - 1 if i - 1 >= 0 else 0
            hyp_w = ''
            for idx in range(startidx, startidx + 2):
                if idx != i:
                    if idx > len(ref_aligned) - 1 or ref_aligned[idx] != '<eps>':
                        continue
                    if idx < i:
                        hyp_w += hyp_aligned[idx] + ' '
                    else:
                        hyp_w += ' ' + hyp_aligned[idx]
                else:
                    hyp_w += hyp_aligned[idx]
            hyp_w = hyp_w.strip()
            hyp_w = hyp_w.replace('<eps>', '')
            d = texterrors_align.lev_distance_str(ref_w, hyp_w)
            oov_count_error += d
    return oov_count_error, oov_count_denom


def read_files(ref_f, hyp_f, isark, rm_events=True):
    rm_list = ['<unk>', '<eps>', '<noise>', '<silence>', '<sil>', '<babble>', '<inaudible>', '<hes>', '<non-speech>']

    utt_to_text_ref = {}
    utts = []
    with open(ref_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                assert utt not in utts, 'There are repeated utterances in reference file! Exiting'
                utts.append(utt)
                if rm_events:
                    utt_to_text_ref[utt] = [w for w in words if not w in rm_list]
                    utt_to_text_ref[utt] = re.subn(r'\[[0-9a-zA-Z/ ]+?\]', '', ' '.join(utt_to_text_ref[utt]))[0].split()

                else:
                    utt_to_text_ref[utt] = [w for w in words]
            else:
                words = line.split()
                if rm_events:
                    utt_to_text_ref[i] =  [w for w in words if not w in rm_list]
                    utt_to_text_ref[i] = re.subn(r'\[[0-9a-zA-Z/ ]+?\]', '', ' '.join(utt_to_text_ref[i]))[0].split()
                else:
                    utt_to_text_ref[i] =  [w for w in words]
                utts.append(i)

    utt_to_text_hyp = {}
    with open(hyp_f) as fh:
        for i, line in enumerate(fh):
            if isark:
                utt, *words = line.split()
                if rm_events:
                    utt_to_text_hyp[utt] =  [w for w in words if not w in rm_list]
                    utt_to_text_hyp[utt] = re.subn(r'\[[0-9a-zA-Z/ ]+?\]', '', ' '.join(utt_to_text_hyp[utt]))[0].split()
                else:
                    utt_to_text_hyp[utt] =  [w for w in words]
            else:
                words = line.split()
                if rm_events:
                    utt_to_text_hyp[i] =  [w for w in words if not w in rm_list]
                    utt_to_text_hyp[i] = re.subn(r'\[[0-9a-zA-Z/ ]+?\]', '', ' '.join(utt_to_text_hyp[i]))[0].split()
                else:
                    utt_to_text_hyp[i] =  [w for w in words]
    return utt_to_text_ref, utt_to_text_hyp, utts


def read_ctm_files(ref_f, hyp_f):
    """ Assumes first field is utt and last three fields are word, time, duration """
    def read_ctm_file(f):
        utt_to_wordtimes = defaultdict(list)
        current_utt = None
        with open(f) as fh:
            for line in fh:
                utt, *_, time, dur, word = line.split()
                time = float(time)
                dur = float(dur)
                utt_to_wordtimes[utt].append((word, time, dur,))
        return utt_to_wordtimes
    utt_to_ref = read_ctm_file(ref_f)
    utt_to_hyp = read_ctm_file(hyp_f)
    utts = list(utt_to_ref.keys())
    return utt_to_ref, utt_to_hyp, utts


def process_files(ref_f, hyp_f, outf, cer=False, count=10, oov_set=None, debug=False,
                  use_chardiff=True, isark=False, skip_detailed=False, insert_tok='<eps>', keywords_list_f='', group_keywords_list_f='',
                  not_score_end=False, no_freq_sort=False, phrase_f='', phrase_list_f='', isctm=False, utt_group_map_f='',class_list=''):


    keywords = set()
    group_keywords={}
    if keywords_list_f:
        for line in open(keywords_list_f):
            assert len(line.split()) == 1, 'A keyword must be a single word!'
            keywords.add(line.strip())
    elif group_keywords_list_f:
        for line in open(group_keywords_list_f):
            assert len(line.split()) == 2, 'Each keyword line must be in <group-name> <word-name> format!'
            line_parts= line.strip().split()
            if not line_parts[0] in group_keywords:
                group_keywords[line_parts[0]] = set()
            group_keywords[line_parts[0]].add(line_parts[1])
    
    rm_events=True
    if keywords or group_keywords:
        rm_events=False

    w2c_dic={}
    c2w_dic={}
    if class_list:
        for line in open(class_list):
            line_parts = line.strip().split()
            assert len(line_parts) > 1, 'Format of class_list file is not correct!'
            w2c_dic[' '.join(line_parts[1:])] = line_parts[0]
            if not line_parts[0] in c2w_dic:
                c2w_dic[line_parts[0]] = list()
            c2w_dic[line_parts[0]].append(' '.join(line_parts[1:]))

    if not isctm:
        utt_to_text_ref, utt_to_text_hyp, utts = read_files(ref_f, hyp_f, isark, rm_events)
    else:
        utt_to_text_ref, utt_to_text_hyp, utts = read_ctm_files(ref_f, hyp_f)
    
    if w2c_dic:
        for utt in utts:
            for ref_hyp in [utt_to_text_ref,utt_to_text_hyp]:
                words = ref_hyp[utt]
                end_idx = len(words)-3
                j=0
                while j < end_idx: 
                    if words[j]+' '+words[j+1]+' '+words[j+2]+' '+words[j+3] in w2c_dic:
                        words[j]=w2c_dic[words[j]+' '+words[j+1]+' '+words[j+2]+' '+words[j+3]]
                        del words[j+1]
                        del words[j+1]
                        del words[j+1]
                        end_idx-=3
                    j+=1

                end_idx = len(words)-2
                j=0
                while j < end_idx: 
                    if words[j]+' '+words[j+1]+' '+words[j+2] in w2c_dic:
                        words[j]=w2c_dic[words[j]+' '+words[j+1]+' '+words[j+2]]
                        del words[j+1]
                        del words[j+1]
                        end_idx-=2
                    j+=1
                end_idx = len(words)-1
                j=0

                while j < end_idx: 
                    if words[j]+' '+words[j+1] in w2c_dic:
                        words[j]=w2c_dic[words[j]+' '+words[j+1]]
                        del words[j+1]
                        end_idx-=1
                    j+=1
                for j in range(len(words)):
                    if words[j] in w2c_dic:
                        words[j]=w2c_dic[words[j]]
                    elif not words[j] in c2w_dic:
                        words[j]='NO_CLASS'
                ref_hyp[utt] = words

                                             
       

    utt2phrase = {}
    utt2phrase_list={}
    

    if phrase_f:
        for line in open(phrase_f):
            utt_words = line.split()
            if len(utt_words) > 1:
                utt2phrase[utt_words[0]] = utt_words[1:]
            else:
                utt2phrase[utt_words[0]] = []
    if phrase_list_f:
        phrase_stats={}
        for line in open(phrase_list_f):
            ph_list=line.strip().split('\t')
            assert len(ph_list[0].split()) == 1, 'Format of phrase_list_f file is not correct! Expected <utt><tab><entity1_tag><space><entity1_words><tab><entity2_tag><space><entity2_words>...'
            utt2phrase_list[ph_list[0]] = []
            if len(ph_list) > 1:
                for k in range(1,len(ph_list)):
                    words=ph_list[k].split()
                    utt2phrase_list[ph_list[0]].append((words[0], words[1:]))



    if utt_group_map_f:
        utt_group_map = {}
        group_stats = {}
        for line in open(utt_group_map_f):
            uttid, group = line.split(maxsplit=1)
            group = group.strip()
            utt_group_map[uttid] = group
            group_stats[group] = {}
            group_stats[group]['count'] = 0
            group_stats[group]['errors'] = 0
            group_stats[group]['utt'] = 0
            group_stats[group]['serr'] = 0

    if outf:
        fh = open(outf, 'w')
    else:
        import sys; fh = sys.stdout

    # Done reading input, processing.
    oov_count_denom = 0
    oov_count_error = 0
    char_count = 0
    char_error_count = 0
    utt_wrong = 0

    ins = defaultdict(int)
    dels = defaultdict(int)
    subs = defaultdict(int)
    total_count = 0
    word_counts = defaultdict(int)
    c_mat=np.zeros((len(c2w_dic.keys())+1,len(c2w_dic.keys())+1))
    c_mat_dic={}
    for cl in c2w_dic.keys():
        c_mat_dic[cl] = np.zeros((2,2))

    if not skip_detailed:
        fh.write('Per utt details:\n')
    dct_char = {insert_tok: 0, 0: insert_tok}
    for utt in utts:
        ref = utt_to_text_ref[utt]
        if utt2phrase:
            phrase = utt2phrase.get(utt)
            if not phrase:
                continue
            is_contained = any([ref[i: i + len(phrase)] == phrase for i in range(len(ref)-len(phrase) + 1)])
            if not is_contained:
                logger.warning(f'A phrase ({phrase}) does not exist in the reference (uttid: {utt})! The phrase'
                               f' must be contained in the reference text! Will not score.')
                continue
        if utt2phrase_list:
            phrase_list = utt2phrase_list.get(utt)
            if not phrase_list:
                continue
            is_contained_all=True
            for phr_key, phr_words in phrase_list:
                is_contained = any([ref[i: i + len(phr_words)] == phr_words for i in range(len(ref)-len(phr_words) + 1)])
                if not is_contained:
                    logger.warning(f'A phrase ({phr_words}) does not exist in the reference (uttid: {utt})! The phrase'
                                f' must be contained in the reference text! Will not score.')
                    is_contained_all=False
                    break
            if not is_contained_all:
                continue
        hyp = utt_to_text_hyp.get(utt)


        
        if hyp is None:
            logger.warning(f'Missing hypothesis for utterance: {utt}')
            #continue
            hyp = ["empty_string"]

        if len(ref) == 0:
            continue

        if not isctm:
            ref_aligned, hyp_aligned, _ = align_texts(ref, hyp, debug, use_chardiff=use_chardiff)
        else:
            ref_words = [e[0] for e in ref]
            hyp_words = [e[0] for e in hyp]
            ref_times = [e[1] for e in ref]
            hyp_times = [e[1] for e in hyp]
            ref_aligned, hyp_aligned, _ = _align_texts_ctm(ref_words, hyp_words, ref_times,
                hyp_times, use_chardiff, debug, insert_tok)

        if not skip_detailed:
            fh.write(f'{utt}\n')
        if not_score_end:
            last_good_index = -1
            for i, (ref_w, hyp_w,) in enumerate(zip(ref_aligned, hyp_aligned)):
                if ref_w == hyp_w:
                    last_good_index = i
        # Finds phrase in reference. There should be a smarter way lol
        if utt2phrase:
            phrase = utt2phrase[utt]
            if not phrase:
                continue
            start_idx = 0
            word_idx = 0
            ref_offset = 1
            while start_idx < len(ref_aligned):
                if phrase[word_idx] == ref_aligned[start_idx]:
                    found = True
                    for i in range(1, len(phrase)):
                        while ref_aligned[start_idx + ref_offset] == '<eps>':
                            ref_offset += 1
                        if phrase[word_idx + i] != ref_aligned[start_idx + ref_offset]:
                            found = False
                            ref_offset = 1
                            break
                        ref_offset += 1
                    if found:
                       break
                start_idx += 1
                word_idx = 0

            ref_aligned = ref_aligned[start_idx: start_idx + ref_offset]
            hyp_aligned = hyp_aligned[start_idx: start_idx + ref_offset]

        entity_ref_hyp_aligned=[]
        if utt2phrase_list:
            phrase_list = utt2phrase_list[utt]
            if not phrase_list:
                continue
            aligned_ids=[]
            start_idx = 0

            for phr_key, phr_words in phrase_list:
                word_idx = 0
                ref_offset = 1

                while start_idx < len(ref_aligned):
                    if phr_words[word_idx] == ref_aligned[start_idx]:
                        found = True
                        for i in range(1, len(phr_words)):
                            while ref_aligned[start_idx + ref_offset] == '<eps>':
                                ref_offset += 1
                            if phr_words[word_idx + i] != ref_aligned[start_idx + ref_offset]:
                                found = False
                                ref_offset = 1
                                break
                            ref_offset += 1
                        if found:
                            break
                    start_idx += 1
                    word_idx = 0
                tmp_idx=[id for id in range(start_idx, start_idx + ref_offset)]
                phrase_ref=[ref_aligned[id] for id in tmp_idx]
                phrase_hyp=[hyp_aligned[id] for id in tmp_idx]

                entity_ref_hyp_aligned.append((phr_key, phrase_ref, phrase_hyp))
                aligned_ids += tmp_idx
            
            ref_aligned = [ref_aligned[id] for id in aligned_ids]
            hyp_aligned = [hyp_aligned[id] for id in aligned_ids]


        if w2c_dic:
            j=0
            end_idx=len(ref_aligned)
            while j < end_idx: 
                if ref_aligned[j] == '<eps>':
                    if hyp_aligned[j] in c2w_dic.keys():
                        ref_aligned[j]='NO_CLASS'
                    else:
                        del ref_aligned[j]
                        del hyp_aligned[j]
                        end_idx-=1
                        j-=1
                elif hyp_aligned[j] == '<eps>':
                    if ref_aligned[j] in c2w_dic.keys():
                        hyp_aligned[j]='NO_CLASS'
                    else:
                        del ref_aligned[j]
                        del hyp_aligned[j]
                        end_idx-=1
                        j-=1                        

                j+=1
            if len(ref_aligned) == 0:
                continue
            c_labels=[cl for cl in c2w_dic.keys()]
            c_labels_nc= c_labels + ['NO_CLASS']
            utt_cmat = confusion_matrix(ref_aligned, hyp_aligned, labels=c_labels_nc)
            c_mat+=utt_cmat

            utt_cb_cmat = multilabel_confusion_matrix(ref_aligned, hyp_aligned,  labels=c_labels_nc)
            for i in range(len(c_labels_nc)-1):
                c_mat_dic[c_labels_nc[i]]+=utt_cb_cmat[i]

        if  group_keywords and utt_group_map_f:
            keywords=group_keywords[utt_group_map[utt]]

        if keywords:
            aligned_ids=[]
            start_idx = 0
            ref_offset=1
            while start_idx < len(ref_aligned):
                if ref_aligned[start_idx] in keywords:
                    aligned_ids.append(start_idx)
                    temp_idx=[]
                    while start_idx + ref_offset < len(ref_aligned)  and ref_aligned[start_idx + ref_offset] == '<eps>':
                        temp_idx.append(start_idx + ref_offset)
                        ref_offset += 1
                        
                    if start_idx + ref_offset != len(ref_aligned) and not ref_aligned[start_idx + ref_offset] in keywords:
                        temp_idx=[]
                    else:
                        for id in temp_idx:
                            aligned_ids.append(int(id))
                    start_idx+= ref_offset
                    ref_offset = 1
                    continue
                start_idx+=1
            if len(aligned_ids) > 0:
                temp_idx=[]
                add_initial=True
                for j in range(aligned_ids[0]):
                    if ref_aligned[j] != '<eps>' and not ref_aligned[j] in keywords:
                        add_initial==False
                        break
                    temp_idx.append(j)
                if add_initial:
                    aligned_ids = temp_idx + aligned_ids
                aligned_ids = [int(id) for id in aligned_ids]
                ref_aligned = [ref_aligned[id] for id in aligned_ids]
                hyp_aligned = [hyp_aligned[id] for id in aligned_ids]
            else:  # skip utterance that contains no keywords
                continue
        colored_output = []
        error_count = 0
        ref_word_count = 0
        for i, (ref_w, hyp_w,) in enumerate(zip(ref_aligned, hyp_aligned)):  # Counting errors
            if not_score_end and i > last_good_index:
                break
            if ref_w == hyp_w:
                colored_output.append(ref_w)
                word_counts[ref_w] += 1
                ref_word_count += 1
            else:
                error_count += 1
                if ref_w == '<eps>':
                    colored_output.append(colored(hyp_w, 'green'))
                    ins[hyp_w] += 1
                elif hyp_w == '<eps>':
                    colored_output.append(colored(ref_w, 'red'))
                    ref_word_count += 1
                    dels[ref_w] += 1
                    word_counts[ref_w] += 1
                else:
                    ref_word_count += 1
                    colored_output.append(colored(f'{ref_w} > {hyp_w}', 'magenta'))
                    subs[f'{ref_w} > {hyp_w}'] += 1
                    word_counts[ref_w] += 1
        total_count += ref_word_count
        if not skip_detailed:
            for w in colored_output:
                fh.write(f'{w} ')
            fh.write('\n')

        if utt_group_map_f:
            group = utt_group_map[utt]
            group_stats[group]['count'] += ref_word_count
            group_stats[group]['errors'] += error_count
            group_stats[group]['utt'] += 1
            if error_count > 0:
                group_stats[group]['serr'] += 1

        if utt2phrase_list:
            for phr_key, phrase_ref, phrase_hyp in entity_ref_hyp_aligned:
                phrase_error_count = 0
                phrase_ref_word_count = 0
                for i, (ref_w, hyp_w,) in enumerate(zip(phrase_ref, phrase_hyp)):  # Counting errors
                    if ref_w == hyp_w:
                        phrase_ref_word_count += 1
                    else:
                        phrase_error_count += 1
                        if ref_w == '<eps>':
                            pass
                        elif hyp_w == '<eps>':
                            phrase_ref_word_count += 1
                        else:
                            phrase_ref_word_count += 1
                if not phr_key in phrase_stats:
                    phrase_stats[phr_key]={}
                    phrase_stats[phr_key]['count']=0
                    phrase_stats[phr_key]['errors']=0
                    phrase_stats[phr_key]['utt']=0
                    phrase_stats[phr_key]['serr']=0
                phrase_stats[phr_key]['count'] += phrase_ref_word_count
                phrase_stats[phr_key]['errors'] += phrase_error_count
                phrase_stats[phr_key]['utt'] += 1
                if phrase_error_count > 0:
                    phrase_stats[phr_key]['serr'] += 1


        if error_count: utt_wrong += 1

        if cer:  # Calculate CER
            if phrase_f:
                raise NotImplementedError('Implementation for CER of phrases not done.')
            def convert_to_char_list(lst):
                new = []
                for i, word in enumerate(lst):
                    for c in word:
                        new.append(c)
                    if i != len(lst) - 1:
                        new.append(' ')
                return new
            char_ref = convert_to_char_list(ref)
            char_hyp = convert_to_char_list(hyp)

            ref_int, hyp_int = convert_to_int(char_ref, char_hyp, dct_char)
            char_error_count += texterrors_align.lev_distance(ref_int, hyp_int)
            char_count += len(ref_int)

        if oov_set:  # Get OOV CER
            err, cnt = get_oov_cer(ref_aligned, hyp_aligned, oov_set)
            oov_count_error += err
            oov_count_denom += cnt

    # Outputting metrics from gathered statistics.
    if w2c_dic:
        tot_tp=np.trace(c_mat[:-1,:-1])
        tot_tn=c_mat[-1,-1]
        tot_class_accuracy=1.0*(tot_tp+tot_tn)/np.sum(c_mat)
        
        print("Total classification accuracy %.2f \n" % (tot_class_accuracy*100) )
    
        for cl in c2w_dic.keys():
            print("Stats for class " + cl + ":" + os.linesep)
            cur_cmat=c_mat_dic[cl]
            class_accuracy=1.0*(cur_cmat[0,0]+cur_cmat[1,1])/np.sum(cur_cmat)
            precision=1.0*cur_cmat[1,1]/(cur_cmat[1,1]+cur_cmat[0,1])
            recall=1.0*cur_cmat[1,1]/(cur_cmat[1,1]+cur_cmat[1,0])
            print("Accuracy %.2f Precision: %.2f Recal: %.2f\n" % (class_accuracy*100, precision*100, recall*100)) 



    ins_count = sum(ins.values())
    del_count = sum(dels.values())
    sub_count = sum(subs.values())
    wer = (ins_count + del_count + sub_count) / float(total_count)
    if not skip_detailed:
        fh.write('\n')
    fh.write(f'WER: {100.*wer:.1f} (ins {ins_count}, del {del_count}, sub {sub_count} / {total_count})'
             f'\nSER: {100.*utt_wrong / len(utts):.1f}\n')

    if cer:
        cer = char_error_count / float(char_count)
        fh.write(f'CER: {100.*cer:.1f} ({char_error_count} / {char_count})\n')
    if oov_set:
        fh.write(f'OOV CER: {100.*oov_count_error / oov_count_denom:.1f}\n')
    if utt_group_map_f:
        fh.write('Group WERS\tAccuracy\tPhrase length:\n')
        sortedkeys=sorted(group_stats.keys(), key=lambda x:x.lower())
        for group in sortedkeys:
            stats = group_stats[group]
            wer = 100. * (stats['errors'] / float(stats['count']))
            acc = 100. * (1 - stats['serr'] / float(stats['utt']))
            phrase_length = (stats['count'] / float(stats['utt']))
            fh.write(f'{group}\t{wer:.1f}\t{acc:.1f}\t{phrase_length:.1f}\n')
        fh.write('\n')

    if utt2phrase_list:
        fh.write('Entity WERS\tAccuracy\tPhrase length\tNum phrase:\n')
        sortedkeys=sorted(phrase_stats.keys(), key=lambda x:x.lower())
        for group in sortedkeys:
            stats = phrase_stats[group]
            wer = 100. * (stats['errors'] / float(stats['count']))
            acc = 100. * (1 - stats['serr'] / float(stats['utt']))
            phrase_length = (stats['count'] / float(stats['utt']))
            phrase_count=stats['utt']
            fh.write(f'{group}\t{wer:.1f}\t{acc:.1f}\t{phrase_length:.1f}\t{phrase_count:d}\n')
        fh.write('\n')

    if not skip_detailed:
        fh.write(f'\nInsertions:\n')
        for v, c in sorted(ins.items(), key=lambda x: x[1], reverse=True)[:count]:
            fh.write(f'{v}\t{c}\n')
        fh.write('\n')
        fh.write(f'Deletions:\n')
        for v, c in sorted(dels.items(), key=lambda x: (x[1] if no_freq_sort else x[1] / word_counts[x[0]]),
                           reverse=True)[:count]:
            fh.write(f'{v}\t{c}\t{word_counts[v]}\n')
        fh.write('\n')
        fh.write(f'Substitutions:\n')
        for v, c in sorted(subs.items(),
                           key=lambda x: (x[1] if no_freq_sort else x[1] / word_counts[x[0].split('>')[0].strip()]),
                           reverse=True)[:count]:
            ref_w = v.split('>')[0].strip()
            fh.write(f'{v}\t{c}\t{word_counts[ref_w]}\n')
    if outf:
        fh.close()


def main(
    fpath_ref: "Reference text",
    fpath_hyp: "Hypothesis text",
    outf: ('Optional output file') = '',
    oov_list_f: ('List of OOVs', 'option', None) = '',
    isark: ('', 'flag', None)=False,
    isctm: ('', 'flag', None)=False,
    cer: ('', 'flag', None)=False,
    debug: ("Print debug messages", "flag", "d")=False,
    no_chardiff: ("Don't use character lev distance for alignment", 'flag', None) = False,
    skip_detailed: ('No per utterance output', 'flag', 's') = False,
    phrase_f: ('Has per utterance phrase which should be scored against, instead of whole utterance', 'option', None) = '',
    phrase_list_f: ('Has per utterance phrase list which should be scored against, instead of whole utterance', 'option', None) = '',
    keywords_list_f: ('Will filter out non keyword reference words.', 'option', None) = '',
    group_keywords_list_f: ('Will filter out non keyword reference words based on the group.', 'option', None) = '',
    no_freq_sort: ('Turn off sorting del/sub errors by frequency (instead by count)', 'flag', None) = False,
    not_score_end: ('Errors at the end will not be counted', 'flag', None) = False,
    utt_group_map_f: ('Should be a file which maps uttids to group, WER will be output per group',
        'option', '') = '',
    class_to_word: ('List of classes with words for computing the class based accuracy',
        'option', '') = ''):

    oov_set = []
    if oov_list_f:
        with open(oov_list_f) as fh:
            for line in fh:
                oov_set.append(line.split()[0])
        oov_set = set(oov_set)
    process_files(fpath_ref, fpath_hyp, outf, cer, debug=debug, oov_set=oov_set,
                 use_chardiff=not no_chardiff, isark=isark, skip_detailed=skip_detailed,
                 keywords_list_f=keywords_list_f, group_keywords_list_f=group_keywords_list_f, not_score_end=not_score_end,
                 no_freq_sort=no_freq_sort, phrase_f=phrase_f, phrase_list_f=phrase_list_f, isctm=isctm,
                 utt_group_map_f=utt_group_map_f,
                 class_list=class_to_word)


if __name__ == "__main__":
    plac.call(main)
