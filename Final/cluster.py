import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn import metrics

D = pd.read_csv('shot_logs.csv')

# feature = D.iloc[:,[1,5,9,10,11,12,18]]

offensive_id = np.unique(D['player_id'].values)
tot_off = D['player_id'].values
# total_match = []
match = D['MATCHUP'].values

shoot_num = D['SHOT_NUMBER'].values
playdate = D['MATCHUP'].values
points = D['PTS'].values
off_dribbles = D['DRIBBLES'].values
off_pointtype = D['PTS_TYPE'].values
off_touch = D['TOUCH_TIME'].values
off_shootdist = D['SHOT_DIST'].values

feature_mat = np.zeros((offensive_id.shape[0], 6))

for i in range(offensive_id.shape[0]):
    # print(i)
    feature = []

    offnum = offensive_id[i]

    playername_row = np.where(tot_off == offnum)

    off_matchdate = match[playername_row]

    off_shoot = shoot_num[playername_row]

    off_point = points[playername_row]

    nonre_offdate = np.unique(off_matchdate)

    stopnum = off_matchdate.shape[0]
    acccount = 0
    maxshootnum = []
    tot_score = []
    for j in range(stopnum):
        tmpcount = 0
        date = off_matchdate[acccount]
        dateindex = np.where(off_matchdate == date)
        tmpcount = dateindex[0].shape[0]

        maxshootind = np.argmax(off_shoot[dateindex])
        maxshootnum.append(off_shoot[maxshootind])

        score = np.sum(off_point[dateindex])
        tot_score.append(score)

        acccount = acccount + tmpcount

        if acccount == stopnum:
            break

    shootarr = np.sum(np.array(maxshootnum))
    mean_score = np.array(tot_score)
    total_match = nonre_offdate.shape[0]

    # 平均投籃次數的feature
    feature_shoot = shootarr / total_match
    feature.append(feature_shoot)

    # 平均投籃距離
    feature_shootdist = np.sum(off_shootdist[playername_row]) / stopnum
    feature.append(feature_shootdist)

    # 總得分
    # feature_score = np.sum(mean_score) / total_match
    # feature.append(feature_score)

    # 命中率
    # feature_shootrate = np.where(off_point > 0)[0].shape[0] / stopnum
    # feature.append(feature_shootrate)

    # 運球次數
    feature_dribbles = np.sum(off_dribbles[playername_row]) / stopnum
    feature.append(feature_dribbles)

    # 持球時間
    feature_touch = np.sum(off_touch[playername_row]) / stopnum
    feature.append(feature_touch)

    # 得分比率(分為兩分與三分)
    two_points = np.where(off_pointtype[playername_row] == 2)[0].shape[0] / stopnum
    feature.append(two_points)
    three_points = np.where(off_pointtype[playername_row] == 3)[0].shape[0] / stopnum
    feature.append(three_points)

    feature_mat[i] = feature

maxval = np.max(feature_mat, axis=0)
for i in range(feature_mat.shape[1]):
    feature_mat[:, i] = feature_mat[:, i] / maxval[i]

kmeans = KMeans(n_clusters=5, random_state=0).fit(feature_mat)
feature_mat_embed = TSNE(n_iter=3000, learning_rate=150).fit_transform(feature_mat)

shooter_id_mapping = np.load('shooter_id_mapping.npy').item()
cluster_id_name_truth = np.load('cluster_id_name_truth.npy')
cluster_id_name_truth = np.concatenate((cluster_id_name_truth, kmeans.labels_.reshape(281, 1)), axis=1)

label = []
for i in range(kmeans.labels_.shape[0]):
    if kmeans.labels_[i] == 2:
        label.append('C')
    elif kmeans.labels_[i] == 4:
        label.append('PF')
    elif kmeans.labels_[i] == 0:
        label.append('SG')
    elif kmeans.labels_[i] == 1:
        label.append('SF')
    elif kmeans.labels_[i] == 3:
        label.append('PG')
label = np.array(label)


plot_data = pd.DataFrame({
    'X1': feature_mat_embed[:, 0],
    'X2': feature_mat_embed[:, 1],
    'Cluster': label
})

fg = sns.FacetGrid(data=plot_data, hue='Cluster')
fg.map(plt.scatter, 'X1', 'X2').add_legend()
plt.show()

# ==================================================================

labels = kmeans.labels_
print(metrics.silhouette_score(feature_mat, labels, metric='euclidean'))
