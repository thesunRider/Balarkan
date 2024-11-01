import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
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

f=1
r0_min = 0.01
R0 = 0.2
r0_max = 0.3

r1_min = f * 0.001
R1 = f * 0.01
r1_max = 1

c1_min = 50
C1 = 120
c1_max = 1000

r2_min = f * 0.01
R2 = f * 0.01
r2_max = 10

c2_min = 500
C2 = 5000
c2_max = 120000

start_soc_min = 0.01
start_soc = 0.5
start_soc_max = 1

capacity_min = 50
capacity = 71
capacity_max = 78


dx = 0.1
voltage_derv = np.gradient(df_hppc_dsg["Voltage(V)"],dx)
cur_derv = np.gradient(df_hppc_dsg["Current_inv(A)"],dx)

peaks_c, c_ = find_peaks(cur_derv, height=10) 
peaks_c_n ,c_ = find_peaks(-cur_derv, height=10)

peaks_v, v_ = find_peaks(voltage_derv, height=0.01) 
peaks_v_n, v_ = find_peaks(-voltage_derv, height=0.01)

peaks_current = np.sort(np.append(peaks_c , peaks_c_n))
peaks_voltage = np.sort(np.append(peaks_v , peaks_v_n))

df_hppc_dsg["vderv"] = voltage_derv
df_hppc_dsg["cderv"] = cur_derv

fig, ax = plt.subplots()
fig.subplots_adjust(right=0.6)

ax2 = ax.twinx()
ax3 = ax.twinx()
ax4 = ax.twinx()


ax3.spines.right.set_position(("axes", 1.2))
ax4.spines.right.set_position(("axes", 1.5))

ax.plot(df_hppc_dsg["Step(s)"],df_hppc_dsg["Current_inv(A)"],"g-",label="Current")
ax2.plot(df_hppc_dsg["Step(s)"],df_hppc_dsg["vderv"],"r-",label="vderv")
ax3.plot(df_hppc_dsg["Step(s)"],df_hppc_dsg["cderv"],"b-",label="cderv")
ax4.plot(df_hppc_dsg["Step(s)"],df_hppc_dsg["Voltage(V)"],color="aqua",label="Voltage")

#p1 = df_hppc_dsg.plot( x='Step(s)', y=[''],ax=ax,style='b-')
#p2 = df_hppc_dsg.plot( x='Step(s)', y=['Voltage(V)'],ax=ax,style='r-',secondary_y=True) 
#p3 = df_hppc_dsg.plot( x='Step(s)', y=['cderv'],ax=ax2,style='g-')

ax.set_ylabel("Current(A)")
ax2.set_ylabel("vderv")
ax3.set_ylabel("cderv")
ax4.set_ylabel("Voltage")

# right, left, top, bottom
ax3.plot(peaks_current, cur_derv[peaks_current], "x")
ax2.plot(peaks_voltage, voltage_derv[peaks_voltage], "x")

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

res_ary = np.array([])
for voltage_indx in peaks_voltage:
	current_indx = find_nearest(peaks_current,voltage_indx)
	res_ary = np.append(res_ary,1000*voltage_derv[voltage_indx]/cur_derv[current_indx])

print(np.abs(res_ary),np.mean(np.abs(res_ary)))

plt.show()