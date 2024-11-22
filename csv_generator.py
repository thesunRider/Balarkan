import numpy as np
import pandas as pd
import os
from xlsx2csv import Xlsx2csv
dir_panda_cache = "pandas_cache\\"
dir_base = "Cleaned_CSV\\Phase1\\"
dir_hppc = dir_base + "HPPC_Data\\"
dir_ocv = dir_base + "OCV_SOC_Data\\"
dir_real = dir_base + "Real_World_Operational_Data\\"

#first generate some csv files

list_hppc = [os.path.abspath(dir_hppc+"\\"+fileName) for fileName in os.listdir(dir_hppc) if fileName.endswith(".xlsx")]
list_ocv = [os.path.abspath(dir_ocv+"\\"+fileName) for fileName in os.listdir(dir_ocv) if fileName.endswith(".xlsx")]
list_real_1 = [os.path.abspath(dir_real+"Scenario 1\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 1") if fileName.endswith(".xlsx")]
list_real_2 = [os.path.abspath(dir_real+"Scenario 2\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 2") if fileName.endswith(".xlsx")]
list_real_3 = [os.path.abspath(dir_real+"Scenario 3\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 3") if fileName.endswith(".xlsx")]
list_real_4 = [os.path.abspath(dir_real+"Scenario 4\\"+fileName) for fileName in os.listdir(dir_real+"\\Scenario 4") if fileName.endswith(".xlsx")]

all_files = [*list_hppc,*list_ocv,*list_real_1,*list_real_2,*list_real_3,*list_real_4]


#generate cache files
for file_check in all_files:
	Xlsx2csv(file_check, outputencoding="utf-8").convert(file_check[:-5]+".csv")