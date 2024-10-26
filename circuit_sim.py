
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

####################################################################################################


#r# Let define a circuit.
f = 1

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

max_ocv = 3.7
min_ocv = 2.5

circuit = Circuit('BatteryModel')

Vbat_elm = circuit.V('Vb', 'Vbat', circuit.gnd, max_ocv@u_V)
#r# When we add an element to a circuit, we can get a reference to it or ignore it:
IL_elm = circuit.PulseCurrentSource('pulse', 'Vl', circuit.gnd,initial_value=0@u_mA, pulsed_value=150@u_mA, pulse_width=10@u_ms, period=20@u_ms,delay_time=1@u_ms)
R0_elm = circuit.R(0, "Res_node", 'Vl', r0_max@u_Ω)
C1_elm = circuit.C(1, 1, "Res_node", c1_max@u_uF)
R1_elm = circuit.R(1, 1, "Res_node", r1_max@u_Ω)
C2_elm = circuit.C(2, 'Vbat', 1, c2_max@u_uF)
R2_elm = circuit.R(2, 'Vbat', 1, r2_max@u_Ω)

def random_circuit_data():
	Vbat_elm.dc_value = round(random.uniform(min_ocv, max_ocv),2)@u_V
	R0_elm.resistance = round(random.uniform(r0_min, r0_max),2)@u_Ω
	R1_elm.resistance = round(random.uniform(r1_min, r1_max),3)@u_Ω
	R2_elm.resistance = round(random.uniform(r2_min, r2_max),2)@u_Ω
	C1_elm.capacitance = round(random.uniform(c1_min, c1_max),0)@u_uF
	C2_elm.capacitance = round(random.uniform(c2_min, c2_max),0)@u_uF
	return Vbat_elm.dc_value,R0_elm.resistance,R1_elm.resistance,R2_elm.resistance,C1_elm.capacitance,C2_elm.capacitance

#PULSE(0 100m 10u 10u 10u 100u 200u)
#r# and modify it
#C1.capacitance = 10@u_F

vis = False

def show_circuit_data():
	#print(circuit)
	simulator = circuit.simulator(temperature=25, nominal_temperature=25)
	analysis = simulator.transient(step_time=50@u_us, end_time=100@u_ms)

	dx = 0.001
	derv = np.gradient(analysis.Vl, dx) # dy/dx 2nd order accurate

	if vis:
		figure, ax = plt.subplots(figsize=(10, 6))
		ax2 = ax.twinx()
		ax3 = ax2.twinx()

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

	return analysis.Vl,derv

data_Y = np.array([])
data_X = np.array([])

with alive_bar(1000) as bar:
	for i in range(0,1000):
		data_Y = np.append(data_Y,[random_circuit_data()])
		data_X = np.append(data_X,[show_circuit_data()[0]]) #only include vdat
		bar()

