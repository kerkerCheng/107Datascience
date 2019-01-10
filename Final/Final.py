import numpy as np
import pandas as pd


def generate_id_name_dict(arr):
    dic = {}
    for i in range(arr.shape[0]):
        dic[arr[i][1]] = arr[i][0]
    return dic


df = pd.read_csv('shot_logs.csv')
df.loc[df['PTS'] < 0, 'PTS'] = 0
df.loc[df['DRIBBLES'] < 0, 'DRIBBLES'] = 0
df.loc[df['TOUCH_TIME'] <= 0, 'TOUCH_TIME'] = 0.01
df.loc[df['SHOT_DIST'] < 0, 'SHOT_DIST'] = 0.01
df.loc[df['CLOSE_DEF_DIST'] <= 0, 'CLOSE_DEF_DIST'] = 0.01

shooter_df = df[['player_name', 'player_id']]
defender_df = df[['CLOSEST_DEFENDER', 'CLOSEST_DEFENDER_PLAYER_ID']]
shooter_df = shooter_df.drop_duplicates()
defender_df = defender_df.drop_duplicates()
shooter_name_id = generate_id_name_dict(shooter_df.values)
defender_name_id = generate_id_name_dict(defender_df.values)

shooter_ngame = df[['GAME_ID', 'player_id']]
shooter_ngame = shooter_ngame.drop_duplicates()
shooter_ngame = shooter_ngame.groupby('player_id').count()

for index, row in shooter_df.iterrows():
    this_df = df.loc[(df['player_id'] == row['player_id'])]
    fg_count = this_df[this_df['FGM'] == 1].shape[0]
    shooter_df.loc[index, 'FG%'] = fg_count/this_df.shape[0]
    shooter_df.loc[index, 'Game_count'] = shooter_ngame.loc[[row['player_id']]].values[0][0]

    score_ef = []
    offense_strength = []
    this_arr = this_df.values
    for index_this, row_this in this_df.iterrows():
        score_ef.append(row_this['PTS']/((row_this['TOUCH_TIME']+row_this['DRIBBLES'])**0.5))
        offense_strength.append(row_this['FGM']*0.3*row_this['SHOT_DIST']/(row_this['CLOSE_DEF_DIST']))

    shooter_df.loc[index, 'score_ef'] = sum(score_ef)/float(len(score_ef))
    shooter_df.loc[index, 'offense_strength'] = sum(offense_strength)/float(len(offense_strength))

shooter_df['FG_ranking'] = shooter_df['FG%'].rank(ascending=False).astype(int)
shooter_df['GC_ranking'] = shooter_df['Game_count'].rank(ascending=False).astype(int)
shooter_df['sco_ranking'] = shooter_df['score_ef'].rank(ascending=False).astype(int)
shooter_df['off_ranking'] = shooter_df['offense_strength'].rank(ascending=False).astype(int)

for index, row in shooter_df.iterrows():
    shooter_df.loc[index, 'ALL_RANK_SCORE'] = 0.75*row['FG_ranking'] + \
                                              1.0*row['GC_ranking'] + \
                                              2.0*row['sco_ranking'] + \
                                              2.0*row['off_ranking']

shooter_df['ALL_RANK'] = shooter_df['ALL_RANK_SCORE'].rank().astype(int)
result = shooter_df[['player_name', 'player_id', 'ALL_RANK']].sort_values(by=['ALL_RANK'])
