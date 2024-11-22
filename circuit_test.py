import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.patches import Rectangle
import pprint

from filterpy.kalman import KalmanFilter

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


import matplotlib.pyplot as plt
import numpy as np
import random
from alive_progress import alive_bar
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

####################################################################################################


from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import *
from PySpice.Spice.HighLevelElement import PieceWiseLinearCurrentSource
####################################################################################################


end_time = 1000
start_indx = 4050 # specify range of graph to analyse
stop_indx = start_indx + end_time
battery_capacity = 280 *60*60#Ah

cb_val = 1000*1e3
rb_val = 5

c2_val = 23
r2_val = 0.3

c1_val = 105
r1_val = 0.7

r0_val = 0.1

max_ocv = 4.18
step_time = 1
current_scale = 1000

working_db = df_hppc_dsg
time_db = working_db["Step(s)"][start_indx:stop_indx].to_numpy() - start_indx
c_db = working_db["Current_inv(A)"][start_indx:stop_indx].to_numpy()
v_db = working_db["Voltage(V)"][start_indx:stop_indx].to_numpy()
soc_db = working_db["SOC_true(%)"][start_indx:stop_indx].to_numpy()

pwl_ary = []
for indx,i in enumerate(time_db):
	if indx >= end_time:
		break
	#if indx > 0 and indx < len(time_db)-1:
	#	if np.abs(c_db[indx+1] - c_db[indx-1]) < 10:
	#		continue
	pwl_ary.append((i@u_s,-c_db[indx]@u_mA))

print(pwl_ary)
print("Starting sim -----------",len(pwl_ary))

plt.plot(c_db)
plt.show()

#r# Let define a circuit.
circuit = Circuit('BatteryModel')



Vbat_elm = circuit.V('Vb', 'Vbat', circuit.gnd, max_ocv@u_V)


Cb_elm = circuit.C(0, 'Vbat', 1, cb_val@u_F,initial_condition=0)
Rb_elm = circuit.R(0, 1, circuit.gnd, rb_val@u_Ω)

C2_elm = circuit.C(2, 1, 2, c2_val@u_F)
R2_elm = circuit.R(2, 1, 2, r2_val@u_Ω)


C1_elm = circuit.C(1, 2, 3, c1_val@u_F)
R1_elm = circuit.R(1, 2, 3, r1_val@u_Ω)

R0_elm = circuit.R(3, 3, 'Vl', r0_val@u_Ω)

#r# When we add an element to a circuit, we can get a reference to it or ignore it:
IL_elm = circuit.PieceWiseLinearCurrentSource(
    'pwlI0', 'Vl', circuit.gnd,
    values=pwl_ary,
)

print(circuit)
print("sims------------")


#PULSE(0 100m 10u 10u 10u 100u 200u)
#r# and modify it
#C1.capacitance = 10@u_F

vis = True

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
#ssimulator.initial_condition(Vl=interp_V_real[5])
analysis = simulator.transient(start_time=0,step_time=step_time@u_s, end_time=end_time@u_s,use_initial_condition=True)

working_v = v_db[0:end_time]
working_t = time_db[0:end_time]

time_spent = np.arange(0,end_time,step_time)


def downsample(array, npts):
	interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
	downsampled = interpolated(np.linspace(0, len(array), npts))
	return downsampled

interp_V_real = np.interp(time_spent,working_t ,working_v )
#interp_V_real = np.roll(interp_V_real,-2)

interp_V_sim = downsample(np.array(analysis.Vl),len(time_spent))
interp_cur = downsample(np.array((analysis["3"] - analysis["Vl"])/R0_elm.resistance),len(time_spent)) 
interp_soc_real = downsample(np.array(soc_db),len(time_spent))


plt.plot(analysis.Vl)
plt.show()

print("Osizes=",len(analysis.Vl),len(working_v))
print("Nsizes=",len(interp_V_sim),len(interp_V_real))


def get_circuit_data(x_axis,Vb,R0,R1,R2,C1,C2):
	global circuit
	Cb,Rb = x_axis
	Vbat_elm.dc_value = Vb@u_V
	Cb_elm.capacitance = Cb@u_F
	Rb_elm.resistance = Rb@u_Ω

	R0_elm.resistance = R0@u_Ω
	R1_elm.resistance = R1@u_Ω
	R2_elm.resistance = R2@u_Ω
	C1_elm.capacitance = C1@u_F
	C2_elm.capacitance = C2@u_F
	#we are not returning the Vbat value as we are taking data differentials

	simulator = circuit.simulator(temperature=25, nominal_temperature=25)
	#simulator.initial_condition(Vl=interp_V_real[5])
	analysis = simulator.transient(start_time=0,step_time=step_time@u_s, end_time=end_time@u_s,use_initial_condition=True)
	based_val = downsample(np.array(analysis.Vl),len(time_spent))
	return based_val

initial_guess = np.array([max_ocv,r0_val, r1_val,r2_val,c1_val,c2_val])
fitParams, fitCovariances = curve_fit(get_circuit_data, [cb_val, rb_val], interp_V_real, p0=initial_guess,bounds=([2,1e-3,1e-3,1e-3,1,1],[4.5,2,2,2,1e4,1e4]))
perr = np.sqrt(np.diag(fitCovariances))

print("fited=",fitParams,perr)

fited_analysis = get_circuit_data([cb_val, rb_val],*fitParams)

current_integral = np.cumsum( interp_cur)
eta = 1000
soc_sim = interp_soc_real[0] - (eta * current_integral/battery_capacity)

#soc_true = df_ocv["Charging_data_SOC"]
#df_ocv["Charging_data_U"]

################## Simulate the estimated parameters for entire circuit

total_indx = len(working_db["Step(s)"])

end_time = total_indx
start_indx = 0 # specify range of graph to analyse
stop_indx = start_indx + total_indx
battery_capacity = 280 *60*60#Ah

max_ocv = 4.18
step_time = 1

working_db = df_hppc_dsg
time_db = working_db["Step(s)"][start_indx:stop_indx].to_numpy() - start_indx
c_db = working_db["Current_inv(A)"][start_indx:stop_indx].to_numpy()
v_db = working_db["Voltage(V)"][start_indx:stop_indx].to_numpy()
soc_db = working_db["SOC_true(%)"][start_indx:stop_indx].to_numpy()

working_v = v_db[0:end_time]
working_t = time_db[0:end_time]

pwl_ary = []
for indx,i in enumerate(time_db):
	if indx >= end_time:
		break
	if indx > 0 and indx < len(time_db)-1:
		if np.abs(c_db[indx+1] - c_db[indx-1]) < 10:
			continue
	pwl_ary.append((i@u_s,-c_db[indx]@u_mA))

IL_elm.detach()
IL_elm = circuit.PieceWiseLinearCurrentSource(
    'pwlI0', 'Vl', circuit.gnd,
    values=pwl_ary,
)

time_spent = np.arange(0,end_time,step_time)
fited_analysis = get_circuit_data([cb_val, rb_val],*fitParams)
interp_V_real = np.interp(time_spent,working_t ,working_v )
interp_cur = downsample(np.array((analysis["3"] - analysis["Vl"])/R0_elm.resistance),len(time_spent)) 

if vis:
	figure, ax = plt.subplots(figsize=(10, 6))
	ax2 = ax.twinx()

	rspine = ax2.spines['right']
	rspine.set_position(('axes', 1))
	ax2.set_frame_on(True)
	ax2.patch.set_visible(False)
	figure.subplots_adjust(right=0.7)

	ax.plot(time_spent,interp_V_real ,color="red")
	ax.plot(time_spent,fited_analysis ,color="green")
	ax2.plot(time_spent,interp_cur,color="blue")
	#ax2.plot(time_spent,interp_soc_real,color="red")
	#ax2.plot(time_spent,soc_sim,color="green")

	ax.grid()
	ax.set_xlabel('t [s]')
	figure.autofmt_xdate()

	plt.tight_layout()
	plt.show()
