## BALARKAN
Battery Analyser using Large Additive Recurrent Kalman Algorithms for Networked Lithium-Ion Batteries. \
Lithium Ion Battery SOC Estimation from Battery parameters such as:

1. HPPC
2. OCV
3. Current History

### Basic 
Lithium-ion batteries are the backbone of a secure, robust, and sustainable energy system.

Ensuring their stable and efficient operation requires continuous advancements and engineering innovations. One critical aspect for efficient battery operation is the State of Charge (SoC) estimation, vital for optimizing battery performance.

Here’s your chance to contribute to a greener future and make a significant impact by enhancing a key component of modern energy systems. In this competition, you will move beyond laboratory data, gaining access to real industry data to develop a robust SoC estimation algorithm applicable in the real-world battery energy storage industry

### Description
Lithium-ion batteries are the backbone of a secure, robust, and sustainable
energy system.
Ensuring their stable and efficient operation requires continuous advancements and
engineering innovations. One critical aspect for efficient battery operation is the
State of Charge (SoC) estimation, vital for optimizing battery performance.
The competitors are invited to develop a robust algorithm to estimate the lithium
iron phosphate battery state of charge.

The competitors are invited to develop a robust algorithm to estimate the lithium iron phosphate battery state of charge.
The developed algorithm must be implemented using Python with available open-source libraries.

DOWNLOAD HPPC, OCV SOC & REAL OPERATION DATASET (ZIP)


### Data Provided

When the preliminary competition opens, the teams will be able to download the
required data to start building their algorithms. The data are:

1) HPPC Data (Hybrid Pulse Power Characterization): a set of discharge-charge pulses, applied to a battery at different states of charge (SOC) and at a given temperature.
2) OCV SOC Data (Open Circuit Voltage vs. State of Charge): Provide the needed data set to implement relationship between the SOC and OCV
3) Real Operation Data: Define real operational scenarios, including the real SoC values under different load profiles.

Goal:
Develop an algorithm for Optimization: Try to minimize the the error between the
estimated SOC and the real SOc on the real operation data


The evaluation criteria will include:
1)Maximum Absolute Error (MaxAE): Measuring the largest absolute error in state of charge estimation.
2) Root Mean Square Error (RMSE): Assessing the overall accuracy of the algorithm by calculating the square root of the mean squared errors.
3) Calculation Efficiency: Evaluating the computational efficiency and speed of the algorithm.
4) Transient Convergence: Assessing how quickly and accurately the algorithm converges during transient conditions.
5) Documentation and Presentation: Judging the clarity and thorough

### Research
$\text{SoC}(t) = \frac{Q_{\text{remaining}}(t)}{Q_{\text{max}}(t)} \times 100 \, \%$ \
$\text{SOC} = \text{SOC}_0 + \frac{1}{C_N} \int_0^t I_{\text{batt}} \, dt$
There will be internal battery looses and other things that come into play too

### Approach
Create a function from HPPC and OCV data whose input takes current and a timestamps from 0 to until that point of time and outputs SOC . can be trained to optimise with real world data too.


### TODO
- [ ] 11min - Onboarding webinar watch

### Reference

1. https://www.sciencedirect.com/science/article/pii/S1452398124001159