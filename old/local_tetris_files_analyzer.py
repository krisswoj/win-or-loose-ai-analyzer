import os
from datetime import datetime

path = '/Users/krzysztofwojdak/Desktop/winorloose_modified'
tetris_results_path = path + '/tetris_results'
date_format = '%Y.%m.%d %H:%M:%S'
date_format_to_print = '%H_%M_%S__%d_%m_%Y'


def get_datetime(date_string):
    return datetime.strptime(date_string, date_format)


def format_to_print(data):
    return data.strftime(date_format_to_print)


def read_tetris_results(round_end_date):
    tetris_results = open('tetris_times.txt', 'r').readlines()

    for i, val in enumerate(tetris_results):
        file_name = val.split(';')[0]
        tetris_result_date = get_datetime(val.split(';')[1][:-1])

        if (tetris_result_date > round_end_date):
            return tetris_results[i - 1].split(';')[0][8:11]

        if (i == len(tetris_results) - 1 and tetris_result_date < round_end_date):
            return file_name[8:11]

        # if (tetris_result_date > round_end_date):
        #     if (i == 0 or i == len(tetris_results) - 1):
        #         return file_name[8:11]
        #     else:
        #         return tetris_results[i - 1].split(';')[0][8:11]


def temp_meth():
    for filename in os.listdir(tetris_results_path):
        if filename.endswith('.txt'):
            with open(tetris_results_path + '/' + filename, encoding='utf-8', errors='replace') as f:
                tetris_logs = f.readlines()
                indices_start = [i for i, s in enumerate(tetris_logs) if ';START' in s]
                indices_end = [i for i, s in enumerate(tetris_logs) if ';end' in s]

                indices_end_filtered = []
                for i, val in enumerate(indices_end):
                    try:
                        if i + 1 == len(indices_end):
                            indices_end_filtered.append(indices_end[i])
                        elif indices_end[i + 1] - indices_end[i] == 1:
                            indices_end_filtered.append(indices_end[i + 1])
                        else:
                            indices_end_filtered.append(indices_end[i])
                    except:
                        pass

                indices_end_filtered = list(set(indices_end_filtered))
                indices_end_filtered.sort()

                tetris_grouped_results = []

                i = 0
                while i < len(indices_end_filtered):
                    tetris_grouped_results.append(tetris_logs[indices_start[i]:indices_end_filtered[i]])
                    i += 1

                len(tetris_grouped_results)

                for i, result in enumerate(tetris_grouped_results):
                    temp = []

                    first_element = result[0]
                    last_element = result[-1]

                    start_date = get_datetime(first_element.split(';')[0])
                    end_date = get_datetime(last_element.split(';')[0])

                    for e in result:
                        tt = e.split(';')
                        time = get_datetime(tt[0])
                        tt.insert(1, str(int((time - start_date).total_seconds())))
                        temp.append(';'.join(tt))

                    time_diff = int((end_date - start_date).total_seconds())

                    temp.append('########' + '\n')
                    temp.append(str(start_date) + '\n')
                    temp.append(str(end_date) + '\n')
                    temp.append(str(time_diff) + '\n')

                    tempfilename = filename[:-4]

                    tetris_end_date = str(read_tetris_results(end_date))

                    with open('tetris_results_new/' + str(tempfilename[0:3]) + '_' + str(tempfilename[4:6]) + '_' + tetris_end_date + '__' + format_to_print(start_date) + '__' + str(time_diff) + 's.txt', "w") as g:
                        g.writelines(temp)


temp_meth()