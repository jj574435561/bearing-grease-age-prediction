import numpy as np
from scipy.io import savemat

from UH60BearingDataMOD import UH60BearingData

path = 'D:\SeongHyeonHong\Python\ARL_Bearing'
folder = path + '\data'     
bearing = UH60BearingData()
filearray = bearing.get_files(folder)
data_size = 800000 # size of the time series data after preprocessing, this is to maintain consistent data length
num_freqs = int(np.floor(data_size/2))
runtime = np.load(path + '\\' + 'mapping.npy').astype('float32')

file_step = 5
batch_size = 100
batch_prev = 100
quo, mod = divmod(len(filearray)//file_step, batch_size)

for batch in range(quo+1):
# for batch in range(1,2):
    
    if batch < quo:
        batch_size = 100
    else:
        batch_size = mod
        
    data_flag = np.zeros((batch_size, 1), dtype='float32')    
    runtime_age = np.zeros((batch_size, 1), dtype='float32')
    yffAE = np.zeros((batch_size, num_freqs), dtype='float32')
    yffVB = np.zeros((batch_size, num_freqs), dtype='float32')
    freqAE = np.zeros((1, num_freqs), dtype='float32')
    freqVB = np.zeros((1, num_freqs), dtype='float32')
    
    file_init = int(file_step*batch_prev*batch)    
    
    for index in range(batch_size):
        file_name = file_init + int(file_step*index)
        data_flag[index], yffAE[index], yffVB[index], TfreqAE, TfreqVB = bearing.matrix_process(folder, file_name, data_size)
        runtime_age[index] = runtime[file_name]
        if data_flag[index] < 1:
            freqAE = TfreqAE
            freqVB = TfreqVB
    
    FFT_Data = {'flag':data_flag, 'runtime':runtime_age, 'yffAE':yffAE, 'yffVB':yffVB, 'freqAE':freqAE, 'freqVB':freqVB}
    savemat(path + '\\fft_data' + '\FFT_Data_{}.mat'.format(batch), FFT_Data)

