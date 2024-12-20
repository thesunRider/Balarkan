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


#THIS IS THE DATA LOADING SESSION (you need to run the cache generator before this)-----------------

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


###### Determine 2RC model parameters
db_working = df_hppc_dsg

def find_current_interval(data):
	data = np.array(data) - np.mean(data)
	T = 20 #sample spacing (1 Second)
	N = data.shape[0]
	yf = fft(data)
	xf = fftfreq(N, T)[:N//2]
	fft_pos = 2.0/N * np.abs(yf[0:N//2])
	yhat = np.array(savgol_filter(fft_pos, 1000, 2))
	peaks_freq, _ = find_peaks(yhat, height=0.1,width = 100) 
	max_peak_index = peaks_freq[np.argmax(yhat[peaks_freq])]
	#find peak with largest amplitude
	#print("gmax=",max_peak_index)
	event_interval = 1/xf[max_peak_index]

	print("Event Interval=",event_interval)
	#plt.plot(xf[peaks_freq],yhat[peaks_freq],"x")
	#plt.plot(xf,yhat)
	#plt.plot(xf[max_peak_index],yhat[max_peak_index],"x")
	#plt.grid()
	#plt.show()
	return int(event_interval)

current_scale = 1000
curr_diff = 0
margin_lapse = 100
event_lapse_time = find_current_interval(db_working["Current_inv(A)"]) + margin_lapse

voltage_derv = np.gradient(db_working["Voltage(V)"],db_working["Step(s)"])
cur_derv = np.gradient(db_working["Current_inv(A)"]/current_scale,db_working["Step(s)"])

#adjust heights according to the hppc current draw
peaks_c, c_ = find_peaks(cur_derv, height=0.01) 
peaks_c_n ,c_ = find_peaks(-cur_derv, height=0.01)

peaks_v, v_ = find_peaks(voltage_derv, height=0.01) 
peaks_v_n, v_ = find_peaks(-voltage_derv, height=0.01)

peaks_current = np.sort(np.append(peaks_c , peaks_c_n))
peaks_voltage = np.sort(np.append(peaks_v , peaks_v_n))

def categorise_event_double(current_peaks):
	current_peaks = np.sort(current_peaks)
	new_last = len(current_peaks) - (len(current_peaks) % 2)
	events_array = current_peaks[0:new_last].reshape(-1, 2)
	new_events = []
	for i in range(0,len(events_array)-1):
		new_events.append(events_array[i])
		new_events.append([events_array[i][1],events_array[i+1][0]])
	
	new_events.append(events_array[-1])
	new_events = np.array(new_events)
	return new_events


def categorise_event_basedoncurrent(current_peaks):
	current_peaks = np.sort(current_peaks)
	new_last = len(current_peaks) - (len(current_peaks) % 3)
	events_array = current_peaks[0:new_last].reshape(-1, 3)
	new_events = []
	for i in range(0,len(events_array)-1):
		new_events.append(events_array[i])
		new_events.append([events_array[i][1],events_array[i][2],events_array[i+1][0]])

	new_events.append(events_array[-1])
	new_events = np.array(new_events)
	return new_events


def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

def categorise_event(a,thr):
	x = np.sort(a)
	diff = x[1:]-x[:-1]
	gps = np.concatenate([[0],np.cumsum(diff>=thr)])
	events = [x[gps==i] for i in range(gps[-1]+1)]
	result = []

	for arr in events:
		if len(arr) == 1:  # If the array contains only one element
			if result:  # Ensure there's a previous array to append to
				result[-1] = np.append(result[-1], arr)
			else:  # If there's no previous array, start a new one
				result.append(arr)
		else:
			result.append(arr)  # Add the array as it is if it has more than one element

	#filtered_events = [arr for arr in events if len(arr) > 1]
	return result


def fill_nan(A):
	inds = np.arange(A.shape[0])
	good = np.where(np.isfinite(A))[0]
	f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
	B = np.where(np.isfinite(A),A,f(inds))
	return B

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return array[idx]

def find_r0():
	res_ary = np.array([])
	for voltage_indx in peaks_voltage:
		current_indx = find_nearest(peaks_current,voltage_indx)
		res_ary = np.append(res_ary,voltage_derv[voltage_indx]/cur_derv[current_indx])
	return np.abs(res_ary)

def rc2_func(t,r_1,tau_1,r_2,tau_2,c):
	global curr_diff
	i = curr_diff
	return np.abs(r_1) *(1-np.exp(-t/np.abs(tau_1)))*i +  np.abs(r_2) *(1-np.exp(-t/np.abs(tau_2)))*i + c

#create a list of events (an event is a current change)
event_list = categorise_event_basedoncurrent(peaks_current) #categorise_event(peaks_current,event_lapse_time)
two_event_list = categorise_event_double(peaks_current)
#Assume Rb as 35 ohms (typical)
Rb = 35

#find Cb
Cb_array = np.array([])
for i in range(0,len(two_event_list)-1):
	first_indx = find_nearest(peaks_voltage,two_event_list[i][0]) -10
	next_index = find_nearest(peaks_voltage,two_event_list[i+1][0]) -10

	diff_voltage = np.abs((db_working["Voltage(V)"])[first_indx] - (db_working["Voltage(V)"])[next_index])
	current_integral = np.trapezoid( (db_working["Current_inv(A)"]/current_scale)[first_indx : next_index] )
	Cb_array = np.abs(np.append(Cb_array,current_integral/diff_voltage))


#find R0
r0_ary = find_r0()

#find R1+R2 , settling time , c1c2_ary
tau_ary = np.array([])
r1r2_ary = np.array([])
c1c2_ary = np.array([])
vis_on = False

for items in event_list:
	start_index = items[0]
	end_index = items[-1]
	work_range = (db_working["Voltage(V)"])[start_index:end_index]
	mavg = moving_average(work_range,100) #100 is the moving window size here (best for this dataset)
	v_diff = np.abs(np.gradient(mavg,1))
	get_indx_stable = np.where(v_diff == 0)[0]
	if  len(get_indx_stable) == 0:
		continue

	#find settling time for voltage and settled voltage
	last_indx_rep  = np.where(items < get_indx_stable[0]+start_index)[0]
	if len(last_indx_rep) == 0:
		continue
	last_indx = items[last_indx_rep[-1]]
	time_difference = np.abs(get_indx_stable[0]+start_index - last_indx)
	#settling time is 4tau,hence diving by 4
	tau_ary = np.append(tau_ary,time_difference/4)

	peak_start_voltage = db_working["Voltage(V)"][last_indx+10]
	settled_voltage = db_working["Voltage(V)"][get_indx_stable[0]+start_index]
	voltage_diff = peak_start_voltage - settled_voltage
	curr_diff = (db_working["Current_inv(A)"][last_indx - 10] - db_working["Current_inv(A)"][get_indx_stable[0]+start_index])/current_scale
	r1r2_ary = np.append(r1r2_ary,abs(voltage_diff/curr_diff))
	#find c1c2_ary ; tau/(R1+R2/2)
	c1c2_ary = np.append(c1c2_ary,(time_difference/4)/(abs(voltage_diff/curr_diff)/2) )

	if vis_on:
		fig, ax = plt.subplots()
		ax.plot(db_working["Step(s)"],db_working["Voltage(V)"])
		ax.axvline(x = get_indx_stable[0]+start_index, color = 'r', linestyle = '-') 
		ax.axvline(x = last_indx, color = 'r', linestyle = '-') 
		ax.plot(db_working["Step(s)"],db_working["Current_inv(A)"]/1000)
		plt.show()


##fit the curve and find actual values

print("-----------------------")
print("Calculated Intial Params:")
print("R0=",r0_ary)
print("r1r2_ary=",r1r2_ary)
print("c1c2_ary=",c1c2_ary)
print("tauary==",tau_ary)
print("-----------------------")

vis_on = False

parameters_fitted = []
for event_id,items in enumerate(two_event_list):
	start_index = items[0]
	end_index = items[1]
	curr_diff = (db_working["Current_inv(A)"][(items[1])] - db_working["Current_inv(A)"][(items[0])])/1000 #cur_derv[items[0]] #

	#print("start_index",start_index,end_index,items,curr_diff)
	work_range = (db_working["Voltage(V)"])[start_index:end_index]
	number_items = work_range.shape[0]
	mean_voltage = np.mean(work_range)
		
	#rc2_func(TI,r_1,tau_1,r_2,tau_2,c)

	x_axis = np.arange(0,number_items)
	initial_guess = np.array([np.mean(r1r2_ary), np.mean(tau_ary),np.mean(r1r2_ary), np.mean(tau_ary),mean_voltage])
	try:

		if vis_on:
			fig, ax = plt.subplots()
			ax.plot(db_working["Step(s)"],db_working["Voltage(V)"])
			ax.plot(x_axis+ start_index,work_range)

			ax.axvline(x = start_index, color = 'r', linestyle = '-') 
			ax.axvline(x = end_index, color = 'r', linestyle = '-') 
			ax.plot(db_working["Step(s)"],db_working["Current_inv(A)"]/1000)


		fitParams, fitCovariances = curve_fit(rc2_func, x_axis, work_range, p0=initial_guess,bounds =(0,1e8))
		perr = np.sqrt(np.diag(fitCovariances))
		if np.any(perr > 1000):
			#print("TM Error:",event_id,perr)
			if vis_on:
				plt.close(fig)
			continue
		
		if vis_on:
			ax.plot(x_axis+ start_index,rc2_func(x_axis,*fitParams))
			plt.show()

		#print("diffcur=",curr_diff)
		parameters_fitted.append({"id":event_id,"guess":initial_guess,"fits":(fitParams,fitCovariances,perr),"range":items})

	except RuntimeError as e:
		#print("Couldnt fit Event:",event_id)
		continue

	
	
#pprint.pp(parameters_fitted)

db_working["vderv"] = voltage_derv
db_working["cderv"] = cur_derv
#db_working["res"] = r0_ary
#db_working["cap"] = Cb_array




if vis_on:
	fig, ax = plt.subplots()
	fig.subplots_adjust(right=0.6)

	ax2 = ax.twinx()
	ax3 = ax.twinx()
	ax4 = ax.twinx()


	ax3.spines.right.set_position(("axes", 1.2))
	ax4.spines.right.set_position(("axes", 1.5))



	ax.plot(db_working["Step(s)"],db_working["Voltage(V)"],"g-",label="VOLTAGE")
	ax2.plot(db_working["Step(s)"],db_working["vderv"],"r-",label="vderv")

	#plot fited data 
	for fits in parameters_fitted:
		start_index = two_event_list[fits["id"]][0]
		end_index = two_event_list[fits["id"]][1]
		x_axis = np.arange(0,end_index-start_index)
		curr_diff = cur_derv[start_index] # db_working["Current_inv(A)"][end_index] - db_working["Current_inv(A)"][start_index]
		ax.plot(x_axis+start_index,rc2_func(x_axis,*(fits["fits"][0]) ) )

	#ax3.plot(db_working["Step(s)"],db_working["cderv"],"b-",label="cderv")
	#ax4.plot(db_working["Step(s)"],db_working["Current_inv(A)"],color="aqua",label="Capacitance")

	#p1 = db_working.plot( x='Step(s)', y=[''],ax=ax,style='b-')
	#p2 = db_working.plot( x='Step(s)', y=['Voltage(V)'],ax=ax,style='r-',secondary_y=True) 
	#p3 = db_working.plot( x='Step(s)', y=['cderv'],ax=ax2,style='g-')

	ax.set_ylabel("VOLTAGE")
	ax2.set_ylabel("vderv")
	#ax3.set_ylabel("cderv")
	#ax4.set_ylabel("Current")

	# right, left, top, bottom
	#ax3.plot(peaks_current, cur_derv[peaks_current], "x")
	#ax2.plot(peaks_voltage, voltage_derv[peaks_voltage], "x")

	print_seps = False
	if print_seps:
		for i in two_event_list:
			col = np.random.rand(3,)
			ax.axvline(x = i[0], color = 'r', linestyle = '-',c=col) 
			ax.axvline(x = i[-1], color = 'r', linestyle = '-',c=col) 
	#print(np.abs(r0_ary),np.mean(np.abs(r0_ary)))



	#print(items,last_indx)
	#ax2.plot(last_indx,0,"x")
	#ax2.plot(get_indx_stable[0]+start_index,0,"x")


	plt.show()


############# determine soc left
print("Determining SOC")

def reject_outliers(data, m = 2.):
	d = np.abs(data - np.median(data))
	mdev = np.median(d)
	s = d/mdev if mdev else np.zeros(len(d))
	return data[s<m]



tau_1_ary_fit = np.array([])
tau_2_ary_fit = np.array([])
c_1_ary_fit = np.array([])
c_2_ary_fit = np.array([])
r_1_ary_fit = np.array([])
r_2_ary_fit = np.array([])


if vis_on:
	fig, ax = plt.subplots()

#{"id":event_id,"guess":initial_guess,"fits":(fitParams,fitCovariances,perr),"range":items})
for count,items in enumerate(parameters_fitted):
	#rc2_func(TI,r_1,tau_1,r_2,tau_2,c)
	r_1_fit = items["fits"][0][0]
	tau_1_fit = items["fits"][0][1]
	r_2_fit = items["fits"][0][2]
	tau_2_fit = items["fits"][0][3]

	c_1_fit = tau_1_fit/r_1_fit
	c_2_fit = tau_2_fit/r_2_fit
	
	tau_1_ary_fit = np.append(tau_1_ary_fit,tau_1_fit)
	tau_2_ary_fit = np.append(tau_2_ary_fit,tau_2_fit)
	r_1_ary_fit = np.append(r_1_ary_fit,r_1_fit)
	r_2_ary_fit = np.append(r_2_ary_fit,r_2_fit)
	c_1_ary_fit = np.append(c_1_ary_fit,c_1_fit)
	c_2_ary_fit = np.append(c_2_ary_fit,c_2_fit)


#limit the values
tau_1_ary_fit = reject_outliers(tau_1_ary_fit,2)
tau_2_ary_fit = reject_outliers(tau_2_ary_fit,2)

r_1_ary_fit = reject_outliers(r_1_ary_fit,50)
r_2_ary_fit = reject_outliers(r_2_ary_fit,50)

c_1_ary_fit = reject_outliers(c_1_ary_fit,1)
c_2_ary_fit = reject_outliers(c_2_ary_fit,1)

r0_ary = reject_outliers(r0_ary,5)
Cb_array = reject_outliers(Cb_array,5)

cb_estim = np.mean(Cb_array)
r0_estim = np.mean(r0_ary)
rb_estim = Rb


max_ocv = 3.5


print("-----------------------")
print("Fitted Intial Params:")
print("R0=",r0_ary,r0_estim)
print("r1_ary=",r_1_ary_fit,np.mean(r_1_ary_fit))
print("r2_ary=",r_2_ary_fit,np.mean(r_2_ary_fit))
print("c1_ary=",c_1_ary_fit,np.mean(c_1_ary_fit))
print("c2_ary=",c_2_ary_fit,np.mean(c_2_ary_fit))
print("cb_ary=",Cb_array,np.mean(Cb_array))
print("tauary==",tau_1_ary_fit,np.mean(tau_1_ary_fit))
print("-----------------------")


if vis_on:
	#print("FIT tau:",np.sort(tau_1_ary_fit+tau_2_ary_fit))
	#print("main tau:",np.sort(tau_ary))
	#ax.plot(np.arange(0,tau_1_ary_fit.shape[0]),tau_1_ary_fit+tau_2_ary_fit,"x")
	#ax.plot(np.arange(0,tau_ary.shape[0]),tau_ary,"o")

	ax.plot(np.arange(0,Cb_array.shape[0]),Cb_array,"x")
	#ax.plot(np.arange(0,r0_ary.shape[0]),r0_ary,"o")
	plt.show()

#### Simulating the circuit and try to optimise the model parameters


import PySpice.Logging.Logging as Logging
from scipy.interpolate import interp1d
logger = Logging.setup_logging()

####################################################################################################

from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import *
from PySpice.Spice.HighLevelElement import PieceWiseLinearCurrentSource
####################################################################################################


working_db = df_hppc_dsg
start_indx = 4050 # specify range of graph to analyse
stop_indx = start_indx + 1000


cb_val = 50*1e3
rb_val = 40

c2_val = 230
r2_val = 0.03

c1_val = 1050
r1_val = 0.07

r0_val = 0.28

max_ocv = 4.18
end_time = 1000
step_time = 0.1


time_db = working_db["Step(s)"][start_indx:stop_indx].to_numpy() - start_indx
c_db = working_db["Current_inv(A)"][start_indx:stop_indx].to_numpy()
v_db = working_db["Voltage(V)"][start_indx:stop_indx].to_numpy()

pwl_ary = []
for indx,i in enumerate(time_db):
	if indx >= end_time:
		break
	#if indx > 0 and indx < len(time_db)-1:
	#	if np.abs(c_db[indx+1] - c_db[indx-1]) < 10:
	#		continue
	pwl_ary.append((i@u_s,-c_db[indx]@u_mA))

print("Starting sim -----------",len(pwl_ary))

plt.plot(c_db)
plt.show()

#r# Let define a circuit.
circuit = Circuit('BatteryModel')



Vbat_elm = circuit.V('Vb', 'Vbat', circuit.gnd, max_ocv@u_V)


Cb_elm = circuit.C(0, 'Vbat', 1, cb_val@u_F,initial_condition=1)
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

print("sims------------")


#PULSE(0 100m 10u 10u 10u 100u 200u)
#r# and modify it
#C1.capacitance = 10@u_F

vis = True

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
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

plt.plot(analysis.Vl)
plt.show()

print("Osizes=",len(analysis.Vl),len(working_v))
print("Nsizes=",len(interp_V_sim),len(interp_V_real))


def get_circuit_data(x_axis,R1,R2,C1,C2):
	global circuit
	Vb,Rb,Cb,R0 = x_axis
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
	analysis = simulator.transient(start_time=0,step_time=step_time@u_s, end_time=end_time@u_s,use_initial_condition=True)
	based_val = downsample(np.array(analysis.Vl),len(time_spent))
	return based_val - np.mean(based_val)

#Start simulation and correction

initial_guess = np.array([r1_val,r2_val,c1_val,c2_val])
fitParams, fitCovariances = curve_fit(get_circuit_data, [max_ocv,rb_val,cb_val,r0_val], interp_V_real-np.mean(interp_V_real), p0=initial_guess,bounds=([1e-3,1e-3,1,1],[2,2,1e4,1e4]),method="dogbox")
perr = np.sqrt(np.diag(fitCovariances))

print("fited parameters [r1_val,r2_val,c1_val,c2_val]=",fitParams)

fited_analysis = get_circuit_data([max_ocv,rb_val,cb_val,r0_val],*fitParams)

#plot the optimised circuit plot for a event witht he new data
if vis:
	figure, ax = plt.subplots(figsize=(10, 6))
	ax2 = ax.twinx()

	rspine = ax2.spines['right']
	rspine.set_position(('axes', 1))
	ax2.set_frame_on(True)
	ax2.patch.set_visible(False)
	figure.subplots_adjust(right=0.7)

	ax.plot(time_spent,interp_V_sim - np.mean(interp_V_sim),color="red")
	ax.plot(time_spent,interp_V_real - np.mean(interp_V_real),color="green")
	ax.plot(time_spent,fited_analysis,color="blue")
	ax2.plot(time_spent,interp_cur,color="blue")


	ax.grid()
	ax.set_xlabel('t [s]')
	figure.autofmt_xdate()

	plt.tight_layout()
	plt.show()