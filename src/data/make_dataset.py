import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
single_file_acc = pd.read_csv('../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv')

single_file_gcc = pd.read_csv('../../data/raw/MetaMotion/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv')


# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob('../../data/raw/MetaMotion/MetaMotion/*.csv')
len(files)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
file_path = '../../data/raw/MetaMotion/MetaMotion\\'
f = files[0]

par_name = f.split('-')[0].replace(file_path,'')
label = f.split('-')[1]
category = f.split('-')[2].rstrip('1234567890').rstrip('_MetaWear_')

df = pd.read_csv(f)
df['participant'] = par_name
df['label'] = label
df['category'] =  category
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    file_path = '../../data/raw/MetaMotion/MetaMotion\\'

    par_name = f.split('-')[0].replace(file_path,'')
    label = f.split('-')[1]
    category = category = f.split('-')[2].rstrip('1234567890').rstrip('_MetaWear_')
    df = pd.read_csv(f)
    df['participant'] = par_name
    df['label'] = label
    df['category'] =  category
    
    if 'Accelerometer' in f :
        df['set'] = acc_set
        acc_set = acc_set + 1
        acc_df = pd.concat([acc_df,df]) 
        
    if 'Gyroscope' in f :
        df['set'] = gyr_set
        gyr_set = gyr_set + 1
        gyr_df = pd.concat([gyr_df,df])


 
    # 

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------

acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

acc_df = acc_df.drop(['epoch (ms)','time (01:00)','elapsed (s)'],axis=1)
gyr_df = gyr_df.drop(['epoch (ms)','time (01:00)','elapsed (s)'],axis=1)


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

def read_data_from_csv(files) :
    

    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1
    
    file_path = '../../data/raw/MetaMotion/MetaMotion\\'

    for f in files:

        par_name = f.split('-')[0].replace(file_path,'')
        label = f.split('-')[1]
        category =  f.split('-')[2].rstrip('1234567890').rstrip('_MetaWear_')
        df = pd.read_csv(f)
        df['participant'] = par_name
        df['label'] = label
        df['category'] =  category
        
        if 'Accelerometer' in f :
            df['set'] = acc_set
            acc_set = acc_set + 1
            acc_df = pd.concat([acc_df,df]) 
            
        if 'Gyroscope' in f :
            df['set'] = gyr_set
            gyr_set = gyr_set + 1
            gyr_df = pd.concat([gyr_df,df])
            
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

    acc_df = acc_df.drop(['epoch (ms)','time (01:00)','elapsed (s)'],axis=1)
    gyr_df = gyr_df.drop(['epoch (ms)','time (01:00)','elapsed (s)'],axis=1)
    
    return acc_df, gyr_df

files = glob('../../data/raw/MetaMotion/MetaMotion/*.csv')
acc_df, gyr_df = read_data_from_csv(files)
    

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

merged_df = pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1)

merged_df.columns = [
    'acc_x',
    'acc_y',
    'acc_z',
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'participant',
    'label',
    'category',
    'set',
]


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    
    'acc_x' : 'mean',
    'acc_y' : 'mean',
    'acc_z' : 'mean',
    'gyr_x' : 'mean',
    'gyr_y' : 'mean',
    'gyr_z' : 'mean',
    'participant' : 'last',
    'label' : 'last',
    'category' : 'last',
    'set' : 'last',
}

merged_df[:1000].resample(rule='200ms').apply(sampling)  

days = [ g for n , g in merged_df.groupby(pd.Grouper(freq='D'))]

resampled_df = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

resampled_df['set'] = resampled_df['set'].astype('int64')

resampled_df['category'] = resampled_df['category'].str.replace(r'\d', '')

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

resampled_df.to_pickle('../../data/interim/data_processed.pkl')

