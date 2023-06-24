import os
import pandas as pd
import librosa
import numpy as np
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocessing():
    root_folder = "sound_dataset"

    audio_vectors = pd.DataFrame(columns=['Actor', 'Audio type', 'Characteristic Vector'])

    # Traverse through the root folder
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Get the two folders upstream
        parent_folder = os.path.basename(dirpath) #Actor number
        grandparent_folder = os.path.basename(os.path.dirname(dirpath)) #Song or Speech
        
        # Iterate over each file in the current directory
        for filename in filenames:
            # Get the full path of the file
            file_path = os.path.join(dirpath, filename)
            
            y, sr = librosa.load(file_path)
            components = 128 #define components number
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=components)
            # print(mfccs.shape)
            
            audio_vectors.loc[len(audio_vectors.index)] = [parent_folder, grandparent_folder, mfccs]

    #Create new dataframe for selecting components for characteristic vector
    df2 = audio_vectors
    vector_means = np.array([])
    for vector in audio_vectors["Characteristic Vector"]:
        vector_mean = np.mean(vector, axis=1) #Get the mean for each component
        vector_means = np.append(vector_means, vector_mean, axis=0) #Stored the components
    vector_means = vector_means.reshape(-1,128)

    df2 = audio_vectors.drop(["Characteristic Vector"], axis=1)
    for i in range(len(vector_means[0])):
        param_number = i+1
        df2["Param_{}".format(param_number)]=vector_means[:,i] #Assign each component per value in df

    for count, actor in enumerate(np.unique(df2["Actor"])):
        true_false = []
        for value in df2["Actor"]==actor:
            true_false.append(int(value))
        df2["{}".format(actor)]=true_false #Verify if it's actor n

    x_train, x_test, y_train, y_test = train_test_split(df2.iloc[:, 2:130], df2.iloc[:, -24:], test_size = 0.3)

    df2.to_csv('dataset.csv', index=False)

    # not normalized data
    x_train.to_csv('x_train.csv',index=False)
    x_test.to_csv('x_test.csv',index=False)
    y_train.to_csv('y_train.csv',index=False)
    y_test.to_csv('y_test.csv',index=False)

    # normalized data
    scaler = MinMaxScaler()
    train_normalized = scaler.fit_transform(x_train) # all but the first 2 columns which are the labels
    df_train_normalized = pd.DataFrame(train_normalized, columns=x_train.columns)
    test_normalized = scaler.fit_transform(x_test) # all but the first 2 columns which are the labels
    df_test_normalized = pd.DataFrame(test_normalized, columns=x_test.columns)
    df_train_normalized.to_csv('x_train_norm.csv',index=False)
    df_test_normalized.to_csv('x_test_norm.csv',index=False)


cwd = os.getcwd()
dataset_path = os.path.join(cwd, 'dataset.csv')
# train_path = os.path.join(cwd, 'training.csv')
# test_path = os.path.join(cwd, 'testing.csv')

if not os.path.exists(dataset_path):
    preprocessing() # here we get the .csvs

cpp_file = "training.cpp"

# Compile the C++ code
compile_command = f"g++ -o {cpp_file[:-4]} {cpp_file} -std=c++17"
subprocess.run(compile_command, shell=True)

# Execute the compiled C++ program
execute_command = f"./{cpp_file[:-4]}"
subprocess.run(execute_command, shell=True)

