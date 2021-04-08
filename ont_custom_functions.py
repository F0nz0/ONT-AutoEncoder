'''
A Set of functions that can be used on ont_fast5_api.fast5_file.Fast5File objects to extend their functionalities.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def raw_to_pA(f5):
    '''
    Function to transform back from raw signal to pA scale.
    '''
    try:
        raw_unit = f5.get_channel_info()["range"] / f5.get_channel_info()["digitisation"]
        pA_signal = (f5.get_raw_data() + f5.get_channel_info()["offset"]) * raw_unit
        return pA_signal
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e)

def get_events(f5):
    '''
    Function to retrieve eventes related data from fast5 file.
    '''
    try:
        f_events = pd.DataFrame(f5.get_analysis_dataset("Basecall_1D_000/BaseCalled_template", "Events"))
        return f_events
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e)

def get_squiggle(f5):
    '''
    Functions that reconstruct the squiggle from the events table.
    '''
    try:
        squiggle = []
        f_events = get_events(f5)
        for row in f_events.itertuples():
            squiggle += [row.mean for i in range(row.length)]
        return squiggle
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e)

def get_fastq(f5, only_sequence=True):
    '''
    Function to retrieve fastq file related to the fast5 read.
    It is possible to retrieve only the sequence (default setting) or
    to retrieve the whole fastq associated with the quality scores.
    '''
    try:
        fastq = f5.get_analysis_dataset("Basecall_1D_000/BaseCalled_template", "Fastq")
        fastq_sequence = fastq.split("\n")[1]
        if only_sequence == True:
            return fastq_sequence
        elif only_sequence == False:
            return fastq
        else:
            raise Exception("Only True and False values are allowed for 'only_sequence' attribute!")
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e)

def fastq_seq_builder_v1(f5, return_events=False):
    '''
    Function to reconstruct fastq sequence from table of events.
    Version 1: It bases on last elements changed in the context of the model.
    IT MAKES SAME ERRORS! PLEASE USE VESION 2!
    '''
    try:
        f_events = get_events(f5)
        df = f_events.sort_index(ascending=False).query("move > 0")
        sequence = ""
        idx_events = [] # where to store basecalled sequences
        idxs = [] # where to store the index of the f_event table to retrive the corresponding row.
        sequence += df.head(1).mp_state.values[0][::-1]
        last_move = df.head(1).move.values[0]
        for row in df.tail(-1).itertuples():
            if row.p_model_state == row.p_mp_state:
                sequence += row.mp_state[0:last_move][::-1]
                idx_events.append(row.mp_state[0:last_move][::-1])
                idxs.append(row.Index)
            else:
                sequence += (row.model_state[0:last_move][::-1]).lower()
                idx_events.append( (row.model_state[0:last_move][::-1]).lower() )
                idxs.append(row.Index)
            last_move = row.move
            sequence = sequence.replace("T", "U")
            sequence = sequence.replace("t", "u")
        if return_events == True:
            return sequence, (idx_events, idxs)
        elif return_events == False:
            return sequence
        else:
            raise Exception("Only True and False values are allowed for 'only_sequence' attribute!")
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e)

def fastq_seq_builder_v2(f5):
    '''
    Function to reconstruct fastq sequence from table of events.
    Version 2: It bases on the bases probabilities that refers to the base at the center of the context.
    It works pretty good but it can be still improved because of some errors (1 or 2 bases) in 
    the first part of the read. 
    '''
    f_events = get_events(f5)
    df = f_events.query("move > 0")
    sequence = ""
    if df.iloc[0].p_model_state == df.iloc[0].p_mp_state:
        sequence += (df.iloc[0].mp_state)[0:2]
    else:
        sequence += (df.iloc[0].model_state)[0:2]

    for i in df.itertuples():
        mapping_dict = {0:"A", 1:"C", 2:"G", 3:"T"}
        base_probs = np.array([i.p_A, i.p_C, i.p_G, i.p_T])
        if i.p_model_state == i.p_mp_state:
            if i.move == 1:
                sequence += i.mp_state[2]
            elif i.move == 2:
                sequence += i.mp_state[1:3]
        elif i.p_model_state < i.p_mp_state:
            if i.move == 1:
                sequence += i.model_state[2]
            elif i.move == 2:
                sequence += i.model_state[1:3]
        elif i.p_model_state > i.p_mp_state:
            if i.move == 1:
                sequence += i.mp_state[2]
            elif i.move == 2:
                sequence += i.mp_state[1:3]

    if df.iloc[-1].p_model_state == df.iloc[-1].p_mp_state:
        sequence += (df.iloc[-1].mp_state)[-2:]
    else:
        sequence += (df.iloc[-1].model_state)[-2:]

    sequence = sequence.replace("T", "U")
    sequence = sequence[::-1]
    return sequence

def plot_signal(f5, start=None, end=None, plot_squiggle=True, plot_pA_signal=True, 
                base_calls=True, base_calls_text = False, figsize=(20,5)):
    '''
    Function to plot either pA scaled signals and event squiggle.
    '''
    try:
        f5_filename = f5.filename.split("\\")[-1].split(".")[0]
        pA_signal = raw_to_pA(f5)
        squiggle = get_squiggle(f5)
        events = get_events(f5).query("move > 0").start
        events_bases = get_events(f5)
        if start == None:
            start = 0
        if end == None:
            end = len(squiggle)
        plt.figure(figsize=figsize)
        if plot_pA_signal == True:
            plt.plot(np.arange(start, end), pA_signal[start:end], label="Scaled PA Signal", zorder=0)
        if plot_squiggle == True:
            plt.plot(np.arange(start, end), squiggle[start:end], label="Event squiggle", zorder=1)
        if base_calls == True:
            plt.scatter(events[(events > start) & (events < end)],
                        np.array(squiggle)[events[(events > start) & (events < end)]] + np.random.randint(-1,1),
                        color = "black", marker="x", s = 40, label="base-calling", zorder=2)
        if base_calls_text == True:
            plt.ylim(plt.axis()[2]*0.8, plt.axis()[3]*1.5)
            for i in events_bases[(events_bases.start > start) & (events_bases.start +15 < end)].itertuples():
                plt.vlines(i.start, plt.axis()[2], plt.axis()[3], colors="r", linestyles="dotted")
                plt.annotate("Move:"+str(i.move), (i.start + 7.5, plt.axis()[3] * 0.95), ha='center')
                plt.annotate("mp_state:\n"+i.mp_state, (i.start + 7.5, plt.axis()[3] * 0.85), ha='center')
                plt.annotate("model_state:\n"+i.model_state, (i.start + 7.5, plt.axis()[3] * 0.75), ha='center')

        plt.title(f5_filename)
        plt.xlabel("measurements")
        plt.ylabel("Current Signal (pA)")
        plt.legend()
        plt.show()
    except Exception as e:
        print("AN EXCEPTION HAS OCCURRED!\n", e)