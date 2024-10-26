
import matplotlib.pyplot as plt
import numpy as np

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

####################################################################################################


from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit, SubCircuitFactory
from PySpice.Unit import *

####################################################################################################


#r# Let define a circuit.

circuit = Circuit('BatteryModel')

Vbat_elm = circuit.V('Vb', 'Vbat', circuit.gnd, 10@u_V)

#r# When we add an element to a circuit, we can get a reference to it or ignore it:
R0_val = 10

IL_elm = circuit.PulseCurrentSource('pulse', 'Vl', circuit.gnd,initial_value=0@u_mA, pulsed_value=100@u_mA, pulse_width=100@u_us, period=200@u_us,delay_time=10@u_us)
R0_elm = circuit.R(0, "Res_node", 'Vl', R0_val@u_Ω)

C1_elm = circuit.C(1, 1, "Res_node", 1@u_uF)
R1_elm = circuit.R(1, 1, "Res_node", 10@u_Ω)

C2_elm = circuit.C(2, 'Vbat', 1, 1@u_uF)
R2_elm = circuit.R(2, 'Vbat', 1, 10@u_Ω)


#PULSE(0 100m 10u 10u 10u 100u 200u)
#r# and modify it
#C1.capacitance = 10@u_F

print(circuit)
simulator = circuit.simulator(temperature=25, nominal_temperature=25)
#simulator.initial_condition() 
analysis = simulator.transient(step_time=1@u_us, end_time=500@u_us)

figure, ax = plt.subplots(figsize=(20, 10))
ax.grid()
ax.plot(analysis.Vl)
ax.plot( 1000*(analysis.Res_node - analysis.Vl )/R0_val)
ax.set_xlabel('t [us]')
ax.set_ylabel('[V]')
ax.legend(('VLoad [V]', 'Ibat [ma]'), loc=(.8,.8))
plt.tight_layout()
plt.show()