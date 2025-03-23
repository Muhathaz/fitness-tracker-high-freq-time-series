import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans



# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle('../../data/interim/02_outliers_removed_chauvenet.pkl')

predictor_variables = list(df.columns[:6])

plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['lines.linewidth'] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in predictor_variables:
    df[col] = df[col].interpolate()
    

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df['set'] == 25]['acc_x'].plot()

duration = df[df['set'] ==1 ].index[-1] -  df[df['set'] ==1 ].index[0]
duration.seconds

for set in df['set'].unique():
    duration = df[df['set'] == set ].index[-1] -  df[df['set'] == set ].index[0]
    df.loc[(df['set'] == set),"duration"] = duration.seconds
    
duration_df = df.groupby(['category'])['duration'].mean()

duration_df.iloc[0]/5
duration_df.iloc[1]/10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()

lowpass = LowPassFilter()

fs = 1000/200
cutoff = 1.2

df_lowpass = lowpass.low_pass_filter(df_lowpass,'acc_y',fs,cutoff,order=5)

subset_df = df_lowpass[df_lowpass['set'] == 13]

fig,ax = plt.subplots(nrows =2, figsize = (20,10))
subset_df[['acc_y']].plot(ax=ax[0])
subset_df[['acc_y_lowpass']].plot(ax=ax[1])

ax[0].legend(loc = 'upper center', 
            bbox_to_anchor =(0.5,1.15), ncol = 3, fancybox = True, shadow = True)
ax[1].legend(loc = 'upper center', bbox_to_anchor =(0.5,1.15), ncol = 3, fancybox = True, shadow = True)

for col in predictor_variables:
    df_lowpass = lowpass.low_pass_filter(df_lowpass,col,fs,cutoff,order=5)
    df_lowpass[col] = df_lowpass[col + '_lowpass' ]
    del df_lowpass[col + '_lowpass' ]

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

pca_values = PCA.determine_pc_explained_variance(df_pca,predictor_variables)

plt.figure(figsize=(10, 10))
plt.plot(range(1,len(predictor_variables)+ 1), pca_values)
plt.xlabel('Principal Component Number')
plt.ylabel('Explained Variance')
plt.show()

df_pca = PCA.apply_pca(df_pca,predictor_variables,3)
subset_df = df_pca[df_pca['set'] == 13]
subset_df[['pca_1','pca_2','pca_3']].plot()

# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------

df_squared = df_pca.copy()
acc_sq = df_squared['acc_x']**2 + df_squared['acc_y']**2  + df_squared['acc_z']**2 
gyr_sq = df_squared['gyr_x']**2 + df_squared['gyr_y']**2  + df_squared['acc_z']**2

df_squared['acc_r'] = np.sqrt(acc_sq)
df_squared['gyr_r'] = np.sqrt(gyr_sq)

subset_df = df_squared[df_squared['set'] == 13]

subset_df[['acc_r','gyr_r']].plot(subplots = True)
# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
df_temporal.dropna(inplace = True)
NumAbs = NumericalAbstraction()

ws = int(1000/200)
predictor_variables = predictor_variables + ['acc_r','gyr_r']

temporal_list = []
for s in df['set'].unique():
    subset_df = df_temporal[df_temporal['set']== s].copy()
    for var in predictor_variables:
        subset_df = NumAbs.abstract_numerical(subset_df,[var],ws,'mean')
        subset_df = NumAbs.abstract_numerical(subset_df,[var],ws,'std')

        
    temporal_list.append(subset_df)

df_temporal = pd.concat(temporal_list)

subset_df[['gyr_x','gyr_x_temp_mean_ws_5','gyr_x_temp_std_ws_5']].plot()
subset_df[['acc_x','acc_x_temp_mean_ws_5','acc_x_temp_std_ws_5']].plot()



# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------
df_freq = df_temporal.copy()
FreqAbs = FourierTransformation()

fs = int(1000/200) # Time interaval for each recorded observation
ws = int(2800/200) #Avg Len of each repetation
 
df_freq_list = []

for s in df['set'].unique():
    print(f'Running Set {s}')
    subset_df = df_freq[df_freq['set']== s].reset_index().copy()

    subset_df = FreqAbs.abstract_frequency(subset_df,predictor_variables, ws , fs)
    df_freq_list.append(subset_df)
    
df_freq = pd.concat(df_freq_list).set_index('epoch (ms)',drop=True)
df_freq.dropna(inplace=True)
# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
df_freq = df_freq.iloc[::2]
# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()

cluster_columns = ['acc_x','acc_y','acc_z']
k_values = range(2,10)
interias = []
subset_df = df_cluster[cluster_columns]
for k in k_values:
    kmeans = KMeans(n_clusters=k,n_init=20,random_state=13)
    cluster_label = kmeans.fit_predict(subset_df)
    interias.append(kmeans.inertia_)
    
plt.figure(figsize=(10,10))
plt.plot(k_values,interias)
plt.xlabel('K Values')
plt.ylabel('Sum of Squared Distance')
plt.show()

kmeans = KMeans(n_clusters=5, n_init=20, random_state=13)
df_cluster['cluster'] = kmeans.fit_predict(subset_df)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection = '3d')
for c in df_cluster['label'].unique():
    subset_df = df_cluster[df_cluster['cluster'] == c]
    ax.scatter(subset_df['acc_x'],subset_df['acc_y'],subset_df['acc_z'], label = c)
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis') 
plt.legend()
plt.title('Cluster Plot')
plt.show()   
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle('../../data/interim/03_feature_extracted.pkl')
