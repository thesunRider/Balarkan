## BALARKAN
Battery Analyser using Large Additive Recurrent Kalman Algorithms for Networked Lithium-Ion Batteries. 
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

Negative current refers to battery discharge, positive current is battery charging

The evaluation criteria will include:
1. Maximum Absolute Error (MaxAE): Measuring the largest absolute error in state of charge estimation.
2. Root Mean Square Error (RMSE): Assessing the overall accuracy of the algorithm by calculating the square root of the mean squared errors.
3. Calculation Efficiency: Evaluating the computational efficiency and speed of the algorithm.
4. Transient Convergence: Assessing how quickly and accurately the algorithm converges during transient conditions.
5. Documentation and Presentation: Judging the clarity and thorough

### Research
```math
\text{SoC}(t) = \frac{Q_{\text{remaining}}(t)}{Q_{\text{max}}(t)} \times 100 \, \% 
```
The coulmb counting method is as follows:
```math
\text{SOC} = \text{SOC}_0 + \frac{1}{C_N} \int_0^t \text{I}_{\text{batt}} \, dt
```

There will be internal battery looses and other things that come into play too

Battery under test is: 

### Approach
Create a function from HPPC and OCV data whose input takes current and a timestamps from 0 to until that point of time and outputs SOC . can be trained to optimise with real world data too.

```math
\text{SOC} = f(\text{I}(t_0\rightarrow t), \text{SOC}_0, \text{OCV},\text{T})
```

Another approach is to make an ECM of the battery consisting of two RC filters. Then use kalman filter and AI to minimise the loses.
Such an RC filter should look like:

![ECM RC Model](/doc_assets/model.jpg)

In this dynamic model $U_{oc}$ and $U_L$ are the open-circuit and terminal voltage in the circuit, respectively. $R_1$ and $R_2$ are the electrochemical and concentration polarization resistance, and $C_1$ and $C_2$ are the electrochemical and concentration polarization capacitors, respectively. In addition, the ohmic internal resistance is $R_0$, and the current flowing through the voltage source is $I_L$. According to Kirchhoff's second and first circuit laws, we have

```math
\begin{aligned}
 y &= x+2 \\
   &= 2+x
\end{aligned}
```

$U_1$ represents the voltage across $R_1$ and $C_1$, while $U_2$ represents the voltage across $R_2$ and $C_2$. Furthermore, the functional connection of SOC may be used to estimate any parameter in the model. The SOC of the battery has been defined as follows:


For convenience, η, representing Coulomb efficiency is set to 1 in the calculation. Moreover, the capacity of the battery is represented by CN, where SOC0 is the starting battery charge. In addition, for the stated dynamic 2-RC ECM, [SOC U1 U2] has been selected as the state variable. The state space equation may be discretized using Eqs. (3), (4) as follows.
(5)
The observation and state equation of the system are represented by Eq. (5), where Δt represents the sampling period. Moreover, 
 and 
 are the time constants where 
=C2R2 and 
 = C1R1 hold. Additionally, the symbols wk and vk denote process and observation noise. CN denotes the rated capacity of the battery. The current time point is expressed by k, while the future time point is represented by k + 1.

### TODO
- [ ] 11min - Onboarding webinar watch

### Reference

1. https://www.sciencedirect.com/science/article/pii/S1452398124001159
2. https://www.sciencedirect.com/science/article/pii/S2352152X24018905