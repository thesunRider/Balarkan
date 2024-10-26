import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

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



fig, ax = plt.subplots()
ax3 = ax.twinx()
rspine = ax3.spines['right']
rspine.set_position(('axes', 1.15))
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
fig.subplots_adjust(right=0.7)


df_hppc_dsg.plot( x='Step(s)', y=['Current_inv(A)'],ax=ax,style='b-')
df_hppc_dsg.plot( x='Step(s)', y=['Voltage(V)'],ax=ax,style='r-',secondary_y=True) 
df_hppc_dsg.plot( x='Step(s)', y=['SOC_true(%)'],ax=ax3,style='g-') 

ax3.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0], ax3.get_lines()[0]],\
           ['Current','Voltage','SOC LVL'], bbox_to_anchor=(1.5, 0.5))
plt.show()