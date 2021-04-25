import os
import pandas as pd
import re

path = '/Users/krzysztofwojdak/Desktop/winorloose_modified'
exports_modified = path + '/exports_modified'
date_format = '%Y.%m.%d %H:%M:%S'
date_format_to_print = '%H_%M_%S__%d_%m_%Y'

path_tetris_results_new = '/Users/krzysztofwojdak/Desktop/winorloose_modified/tetris_results_new'


def temp_meth():
    bios_data_info = []

    for filename in os.listdir(exports_modified):
        if filename.endswith('.txt'):
            bios = pd.read_csv(exports_modified + '/' + filename)['Czas'][-1:].item()
            bios_data_info.append(filename + ' length: ' + str(bios) + 's\n')

    bios_data_info.sort()

    with open('bios_length.txt', "w") as g:
        g.writelines(bios_data_info)


def merge_times():
    with open(path_tetris_results_new + '/bios_length.txt', 'r') as f:
        bios_length = f.readlines()

    tetris_file_names = []
    for filename in os.listdir(path_tetris_results_new + '/all'):
        tetris_file_names.append(filename)

    temp = []
    for i in bios_length:
        tetris_file_name = [s for s in tetris_file_names if i[0:4] + i[5:11] in s]

        val = 0
        for j in tetris_file_name:
            val += int(re.findall('(?<=)\d+(?=s\.txt)', j)[0])
        temp.append(i[:-1] + ', counted rounds time: ' + str(val) + 's\n')

    temp.sort()
    len(temp)

    with open('bios_length_vs_tetris.txt', "w") as g:
        g.writelines(temp)


merge_times()
# temp_meth()
