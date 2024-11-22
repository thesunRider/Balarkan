import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import interpolate
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import pprint

##THIS LOADS FROM THE CACHE DIRECTORIES
dir_base = "pandas_cache\\"
dir_hppc = dir_base + "HPPC_Data\\"
dir_ocv = dir_base + "OCV_SOC_Data\\"
dir_real = dir_base + "Real_World_Operational_Data\\"

list_hppc = [os.path.abspath(dir_hppc+"\\"+fileName) for fileName in os.listdir(dir_hppc) if fileName.endswith(".parquet")]
list_ocv = [os.path.abspath(dir_ocv+"\\"+fileName) for fileName in os.listdir(dir_ocv) if fileName.endswith(".parquet")]
list_real_1 = [os.path.abspath(dir_real+"Scenario 1\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 1") if fileName.endswith(".parquet")]
list_real_2 = [os.path.abspath(dir_real+"Scenario 2\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 2") if fileName.endswith(".parquet")]
list_real_3 = [os.path.abspath(dir_real+"Scenario 3\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 3") if fileName.endswith(".parquet")]
list_real_4 = [os.path.abspath(dir_real+"Scenario 4\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 4") if fileName.endswith(".parquet")]

#all_files = [*list_hppc,*list_ocv,*list_real_1,*list_real_2,*list_real_3,*list_real_4]

df_hppc_chg  = pd.read_parquet("pandas_cache\\HPPC_Data\\EVE_HPPC_1_25degree_CHG-injectionTemplate.parquet")
df_hppc_dsg  = pd.read_parquet("pandas_cache\\HPPC_Data\\EVE_HPPC_1_25degree_DSG-injectionTemplate.parquet")
df_ocv 		 = pd.read_parquet("pandas_cache\\OCV_SOC_Data\\Cha_Dis_OCV_SOC_Data.parquet")

df_real_1_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 1\\GenerateTestData_S1_DAY0to4.parquet")
df_real_1_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 1\\GenerateTestData_S1_DAY0to4.parquet")

df_real_2_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 2\\GenerateTestData_S2_DAY0to4.parquet")
df_real_2_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 2\\GenerateTestData_S2_DAY4to7.parquet")

df_real_3_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 3\\GenerateTestData_S3_DAY0to4.parquet")
df_real_3_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 3\\GenerateTestData_S3_DAY4to7.parquet")

df_real_4_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 4\\GenerateTestData_S4_DAY0to4.parquet")
df_real_4_04 = pd.read_parquet("pandas_cache\\Real_World_Operational_Data\\Scenario 4\\GenerateTestData_S4_DAY4to7.parquet")

working_db = df_hppc_dsg
plot = working_db["Voltage(V)"].plot(title="DataFrame Plot")

file_name = "dsg_pwl_i.txt"

ary1 = np.array(working_db["Step(s)"])
ary2 =  np.array(-working_db["Current_inv(A)"]/1000)
ary3 =  np.array(-working_db["Voltage(V)"])

stacked_array = np.column_stack((ary1,ary2))
np.savetxt(file_name, stacked_array, fmt="%f", delimiter=" ")


stacked_array = np.column_stack((ary1,ary3))
np.savetxt("dsg_pwl_v.txt", stacked_array, fmt="%f", delimiter=" ")
plt.show()