
import matplotlib.pyplot as plt
import numpy as np
import random
from alive_progress import alive_bar
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
####################################################################################################


from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import *

####################################################################################################


#r# Let define a circuit.
f = 1

r0_min = 0.1
R0 = 0.2
r0_max = 5

r1_min = f * 0.01
R1 = f * 0.4
r1_max = 1

c1_min = 100000
C1 = 12000
c1_max = 800000

r2_min = f * 0.01
R2 = f * 0.01
r2_max = 0.1

c2_min = 10000
C2 = 5000
c2_max = 100000

start_soc_min = 0.01
start_soc = 0.5
start_soc_max = 1

capacity_min = 50
capacity = 71
capacity_max = 78

max_ocv = 3.7
min_ocv = 2.5

circuit = Circuit('BatteryModel')

Vbat_elm = circuit.V('Vb', 'Vbat', circuit.gnd, max_ocv@u_V)
#r# When we add an element to a circuit, we can get a reference to it or ignore it:
IL_elm = circuit.PulseCurrentSource('pulse', 'Vl', circuit.gnd,initial_value=0@u_mA, pulsed_value=300@u_A, pulse_width=100@u_ms, period=500@u_ms,delay_time=1@u_ms)
R0_elm = circuit.R(0, "Res_node", 'Vl', r0_max@u_Ω)
C1_elm = circuit.C(1, 1, "Res_node", c1_max@u_F)
R1_elm = circuit.R(1, 1, "Res_node", r1_max@u_Ω)
C2_elm = circuit.C(2, 'Vbat', 1, c2_max@u_F)
R2_elm = circuit.R(2, 'Vbat', 1, r2_max@u_Ω)

def random_circuit_data():
	Vbat_elm.dc_value = round(random.uniform(min_ocv, max_ocv),2)@u_V
	R0_elm.resistance = round(random.uniform(r0_min, r0_max),2)@u_Ω
	R1_elm.resistance = round(random.uniform(r1_min, r1_max),3)@u_Ω
	R2_elm.resistance = round(random.uniform(r2_min, r2_max),2)@u_Ω
	C1_elm.capacitance = round(random.uniform(c1_min, c1_max),0)@u_F
	C2_elm.capacitance = round(random.uniform(c2_min, c2_max),0)@u_F
	#we are not returning the Vbat value as we are taking data differentials
	return np.array([0+R0_elm.resistance,0+R1_elm.resistance,0+R2_elm.resistance,0+C1_elm.capacitance,0+C2_elm.capacitance])

#PULSE(0 100m 10u 10u 10u 100u 200u)
#r# and modify it
#C1.capacitance = 10@u_F

vis = True
series_size = 500
scaler = StandardScaler()

def show_circuit_data():
	#print(circuit)
	simulator = circuit.simulator(temperature=25, nominal_temperature=25)
	analysis = simulator.transient(start_time=1@u_s,step_time=1@u_ms, end_time=2@u_s)

	dx = 0.001
	derv = np.gradient(analysis.Vl, dx) # dy/dx 2nd order accurate

	if vis:
		figure, ax = plt.subplots(figsize=(10, 6))
		ax2 = ax.twinx()
		ax3 = ax.twinx()

		rspine = ax2.spines['right']
		rspine.set_position(('axes', 1))
		ax2.set_frame_on(True)
		ax2.patch.set_visible(False)

		rspine = ax3.spines['right']
		rspine.set_position(('axes', 1.15))
		ax3.set_frame_on(True)
		ax3.patch.set_visible(False)
		figure.subplots_adjust(right=0.7)


		ax.plot(analysis.Vl,color="red")
		ax2.plot( 1000*(analysis.Res_node - analysis.Vl )/R0_elm.resistance,color="blue")
		ax3.plot(derv,color="green")

		ax.grid()
		ax.set_xlabel('t [us]')
		
		ax.set_ylabel("Voltage [V]", color="red", fontsize=14)
		ax.tick_params(axis="y", labelcolor="red")

		ax2.set_ylabel("Current [mA]", color="blue", fontsize=14)
		ax2.tick_params(axis="y", labelcolor="blue")

		figure.autofmt_xdate()

		plt.tight_layout()
		plt.show()

	ret_voltage = np.array(np.array(analysis.Vl[:series_size]))
	ret_derv =  np.array(np.array(derv[:series_size]))
	#print(ret_voltage.shape[0],ret_derv.shape[0])
	assert  (ret_voltage.shape[0] == series_size) and (ret_derv.shape[0] == series_size) , "Not correct size"
	
	#data_Y = scaler.fit_transform(data_Y)
	ret_voltage = ret_voltage - ret_voltage.mean()
	ret_derv = ret_derv - ret_derv.mean()
	return ret_voltage,ret_derv

data_Y = np.array([random_circuit_data()])
data_X = np.array([show_circuit_data()[1]])



sample_size = 100
with alive_bar(sample_size) as bar:
	for i in range(0,sample_size):
		data_Y = np.append(data_Y,[random_circuit_data()],axis=0)
		data_X = np.append(data_X,[show_circuit_data()[0]],axis=0) #only include vdat
		bar()

print("SHAPE x:",data_X.shape)
print("SHAPE y:",data_Y.shape)

#print(data_X)
#print()
#print(data_Y)

##AI setup

model = Sequential([
    LSTM(64, activation='relu', input_shape=(series_size, 1), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    Dense(15),
    LeakyReLU(alpha=0.1),
    Dense(data_Y.shape[1])  # Output layer with 5 units for the 5 constants
])

optimizer = keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2, random_state=21)

print("check nanX:",np.any(np.isnan(X_train)))
print("check nanY:",np.any(np.isnan(y_train)))

#print(model.summary())
#data_in = np.reshape(data_X, (data_X.shape[0], data_X.shape[1], 1))
#new_dat = np.reshape(X_train,(X_train.shape[0], X_train.shape[1], 1))
#print("new dat shape:",new_dat.shape)
history = model.fit(np.array(X_train), np.array(y_train), epochs=1000,  verbose=1, validation_data=(np.array(X_test), np.array(y_test)))
