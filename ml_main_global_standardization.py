from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.preprocessing import StandardScaler

PATH_WITH_CSVS = '/Users/krzysztofwojdak/Documents/win-or-loose-ai-analyzer/rnn_preprocess_data_csv/'
PATH_WITH_CSVS_NEW = '/Users/krzysztofwojdak/Documents/win-or-loose-ai-analyzer/rnn_preprocess_data_csv_new/'


def read_files(path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.csv')]


csv_file_name_list = read_files(PATH_WITH_CSVS)


def player_result_converter(draw, lose, win):
    if draw == 1:
        return 0
    if lose == 1:
        return -1
    if win == 1:
        return 1


def data_normalization():
    dfs = []

    for csv in csv_file_name_list:
        df = pd.read_csv(PATH_WITH_CSVS + csv, sep=',', header=0)
        # column_names = df.columns.values[1:-6]
        df['p1_result'] = df.apply(lambda x: player_result_converter(x['Draw-1'], x['Lose-1'], x['Win-1']), axis=1)
        df.drop(['Draw-1', 'Lose-1', 'Win-1', 'Draw-2', 'Lose-2', 'Win-2'], axis=1, inplace=True)
        dfs.append(df)
        # df.to_csv(PATH_WITH_CSVS_NEW + csv, sep=',', index=False)

    pd_concat = pd.concat(dfs)
    droped = pd_concat.drop(['Czas', 'p1_result'], axis=1)
    scaler = StandardScaler().fit(droped)

    for csv in csv_file_name_list:
        df = pd.read_csv(PATH_WITH_CSVS + csv, sep=',', header=0)
        df['p1_result'] = df.apply(lambda x: player_result_converter(x['Draw-1'], x['Lose-1'], x['Win-1']), axis=1)
        df.drop(['Draw-1', 'Lose-1', 'Win-1', 'Draw-2', 'Lose-2', 'Win-2'], axis=1, inplace=True)
        df_droped = df.drop(['Czas', 'p1_result'], axis=1)

        df_transformed = scaler.transform(df_droped)
        df_transformed_new_df = pd.DataFrame(data=df_transformed, columns=df_droped.columns)
        df_transformed_new_df.insert(0, 'Czas', df['Czas'])
        df_transformed_new_df['p1_result'] = df['p1_result']

        df_transformed_new_df.to_csv(PATH_WITH_CSVS_NEW + csv, sep=',', index=False)


data_normalization()
