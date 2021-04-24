# %%

import numpy as np
import pandas as pd
import re
from os import listdir
from os.path import isfile, join
from datetime import datetime

# Make numpy printouts easier to read.
FULL_PATH = '/Users/krzysztofwojdak/Documents/win-or-loose-ai-analyzer/'
TETRIS_DATA_PATH = 'data/tetris'
BIOSIGNALS_DATA_PATH = 'data/biosignals'
date_format = '%Y-%m-%d %H:%M:%S'
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)


# %%

def get_biosignal_in_range(biosignals, first_range, second_range):
    return biosignals.loc[(first_range + 2 <= biosignals['Czas']) & (biosignals['Czas'] < second_range + 2)]


# %%

def read_files(path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.txt')]


def read_biosignal_data(biosignal_path):
    biosignals = pd.read_csv(FULL_PATH + biosignal_path, sep=',', error_bad_lines=False, index_col=False,
                             dtype='unicode').iloc[1:, :11]
    for columnName in biosignals.columns.values:
        biosignals[columnName] = pd.to_numeric(biosignals[columnName], errors='coerce')

    return {biosignal_path[-15:-4]: biosignals}


def read_tetris_data(tetris_data_path):
    with open(FULL_PATH + tetris_data_path) as f:
        full_lines = f.readlines()
        lines = full_lines[1:-4]

    dataset = pd.DataFrame([sub.rstrip().split(";") for sub in lines], columns=['time', 'second', 'player', 'move'])
    dataset['second'] = pd.to_numeric(dataset['second'], errors='coerce')
    dataset['player'] = pd.to_numeric(dataset['player'], errors='coerce')

    temp_tetris_name = tetris_data_path[12:][:32]
    tetris_file_name = temp_tetris_name[:4] + 'R' + temp_tetris_name[4:]

    return {tetris_file_name: {
        'times': [full_lines[-3].rstrip(), full_lines[-2].rstrip(), int(full_lines[-1].rstrip())],
        'p1': get_first_player_tetris_state(dataset),
        'p2': get_second_player_tetris_state(dataset)
    }}


# %%

LINES_PATTERN = "(?<=lines\()[^)]*(?=\))"


def get_first_player_tetris_state(data):
    max_time = data.tail(1)['second'].item()
    tetris_result = []
    first_player = 1
    first_player_lines = 0
    second_player_lines = 0
    i = 0

    while i < max_time:
        temp_data = data.loc[data['second'] == i]
        if temp_data.empty:
            method_name(first_player_lines, second_player_lines, tetris_result)
            i += 1
            continue
        for index, row in temp_data.iterrows():
            if re.search(LINES_PATTERN, row['move']):
                line_number = int(re.findall(LINES_PATTERN, row['move'])[0])
                if row['player'] == first_player:
                    first_player_lines = line_number
                else:
                    second_player_lines = line_number
        method_name(first_player_lines, second_player_lines, tetris_result)
        i += 1
    return tetris_result


def method_name(first_player_lines, second_player_lines, tetris_result):
    if first_player_lines > second_player_lines:
        tetris_result.append(1)
    elif first_player_lines < second_player_lines:
        tetris_result.append(-1)
    else:
        tetris_result.append(0)


def get_second_player_tetris_state(data):
    first_player = get_first_player_tetris_state(data)
    return [state * -1 for state in first_player]


# %%

def merge_p1_biosignals_with_tetris_results(biosignals, tetris_results, time_start):
    verify_biosignals_with_tetris_time(biosignals, tetris_results)
    return insert_p1_tetris_result(biosignals, tetris_results, time_start)


def merge_p2_biosignals_with_tetris_results(biosignals, tetris_results, time_start):
    verify_biosignals_with_tetris_time(biosignals, tetris_results)
    return insert_p2_tetris_result(biosignals, tetris_results, time_start)


def merge_both_biosignals_with_tetris_results(biosignals, tetris_results, time_start):
    verify_biosignals_with_tetris_time(biosignals, tetris_results)
    return insert_p1p2_tetris_result(biosignals, tetris_results, time_start)


def verify_biosignals_with_tetris_time(biosignals, tetris_results):
    tetris_max_time = tetris_results.iloc[-1, 0].item()
    bio_start_time = biosignals.head(1)['Czas'].item()
    bio_end_time = biosignals.tail(1)['Czas'].item()

    if 0 > (tetris_max_time - (bio_end_time - bio_start_time)) or (
            tetris_max_time - (bio_end_time - bio_start_time)) > 1:
        raise ValueError('Tetris times are not compatible with biosignals times')


# %%

def insert_p1p2_tetris_result(biosignals, tetris_result, time_start):
    first_player_tetris_state = get_first_player_tetris_state(tetris_result)
    second_player_tetris_state = get_second_player_tetris_state(tetris_result)
    first_player = insert_tetris_signals(biosignals, first_player_tetris_state, time_start, '-1')
    return insert_tetris_signals(first_player, second_player_tetris_state, time_start, '-2')


def insert_p1_tetris_result(biosignals, tetris_result, time_start):
    biosignals = biosignals.drop(['B: BVP2', 'D: EMG2', 'H: Thor Resp', 'I: Temp2', 'SC-Pro/Flex - 1J'], axis=1)
    first_player_tetris_state = get_first_player_tetris_state(tetris_result)
    return insert_tetris_signals(biosignals, first_player_tetris_state, time_start, '')


def insert_p2_tetris_result(biosignals, tetris_result, time_start):
    biosignals = biosignals.drop(['A: BVP1', 'C: EMG1', 'E: Skin Cond', 'F: Temp1', 'G: Abd Resp'], axis=1)
    second_player_tetris_state = get_second_player_tetris_state(tetris_result)
    return insert_tetris_signals(biosignals, second_player_tetris_state, time_start, '')


def insert_tetris_signals(biosignals, tetris_state, time_start, tetris_column_sufix):
    merged_tetris_result_with_biosignals = biosignals.apply(lambda x: tetris_state[int(x['Czas']) - 2 - time_start],
                                                            axis=1)
    biosignals = biosignals.assign(tetris=merged_tetris_result_with_biosignals.values)
    biosignals['tetris'] = biosignals['tetris'].map(
        {-1: 'Lose' + tetris_column_sufix, 0: 'Draw' + tetris_column_sufix, 1: 'Win' + tetris_column_sufix})
    return pd.get_dummies(biosignals, prefix='', prefix_sep='')


# %%

# def get_players_results(first_measurement_start_time, first_measurement_end_time, second_measurement_start_time, second_measurement_end_time, folder_name):
#     biosignals = read_biosignal_data(folder_name + '/biosignals.txt')
#     r1_ranged_biosignals = get_biosignal_in_range(biosignals, first_measurement_start_time, first_measurement_end_time)
#     r2_ranged_biosignals = get_biosignal_in_range(biosignals, second_measurement_start_time, second_measurement_end_time)
#
#     first_round_tetris_data = read_tetris_data(folder_name + '/tetris_first_round.txt')
#     second_round_tetris_data = read_tetris_data(folder_name + '/tetris_second_round.txt')
#
#     p1p2r1 = merge_both_biosignals_with_tetris_results(r1_ranged_biosignals, first_round_tetris_data, first_measurement_start_time)
#     p1p2r2 = merge_both_biosignals_with_tetris_results(r2_ranged_biosignals, second_round_tetris_data, second_measurement_start_time)
#     p1r1 = merge_p1_biosignals_with_tetris_results(r1_ranged_biosignals, first_round_tetris_data, first_measurement_start_time)
#     p2r1 = merge_p2_biosignals_with_tetris_results(r1_ranged_biosignals, first_round_tetris_data, first_measurement_start_time)
#     p1r2 = merge_p1_biosignals_with_tetris_results(r2_ranged_biosignals, second_round_tetris_data, second_measurement_start_time)
#     p2r2 = merge_p2_biosignals_with_tetris_results(r2_ranged_biosignals, second_round_tetris_data, second_measurement_start_time)
#
#     return {
#         'biosignals' : biosignals,
#         'p1p2r1':p1p2r1,
#         'p1p2r2':p1p2r2,
#         'p1r1': p1r1,
#         'p2r1': p2r1,
#         'p1r2': p1r2,
#         'p2r2': p2r2,
#     }

# %%

def take_second(elem):
    return elem['times'][0]


def get_datetime(date_string):
    return datetime.strptime(date_string, date_format)


def get_time_range(tetris_data):
    initial_time = 0
    end_time = 0
    time_range = []

    for e in tetris_data:
        times = e['times']

        if end_time != 0:
            initial_time += int((get_datetime(times[0]) - end_time).total_seconds())

        time_range.append([initial_time, initial_time + times[2]])
        initial_time += times[2]
        end_time = get_datetime(times[1])

    return time_range


def get_players_results():
    biosignals_data = read_files(BIOSIGNALS_DATA_PATH)
    tetris_data = read_files(TETRIS_DATA_PATH)

    biosignals_data = list(map(lambda x: read_biosignal_data(BIOSIGNALS_DATA_PATH + '/' + x), biosignals_data))
    biosignals_data = {k: v for d in biosignals_data for k, v in d.items()}
    tetris_data = list(map(lambda x: read_tetris_data(TETRIS_DATA_PATH + '/' + x), tetris_data))
    tetris_data = {k: v for d in tetris_data for k, v in d.items()}

    biosignals_with_tetris = {}
    for bio_key in biosignals_data:
        _tetris_data = [value for key, value in tetris_data.items() if bio_key in key]
        _tetris_data.sort(key=lambda x: x['times'][0])

        temp = {
            'biosignals': biosignals_data[bio_key],
            'tetris': _tetris_data,
            'tetris_time_range': get_time_range(_tetris_data)
        }
        biosignals_with_tetris.update({bio_key: temp})

    del biosignals_data
    del temp
    del tetris_data

    biosignals_with_tetris = dict(sorted(biosignals_with_tetris.items()))

    return biosignals_with_tetris


get_players_results()
