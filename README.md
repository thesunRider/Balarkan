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

### Research & Approach
```math
\text{SoC}(t) = \frac{Q_{\text{remaining}}(t)}{Q_{\text{max}}(t)} \times 100 \, \% 
```

There will be internal battery looses and other things that come into play too

Battery under test is: Li-Ion

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
\text{U}_L &= U_{\text{oc}} - I_L R_0 - U_1 - U_2 \\
\frac{dU_1}{dt} &= \frac{I_L}{C_1} - \frac{U_1}{C_1 R_1} \\
\frac{dU_2}{dt} &= \frac{I_L}{C_2} - \frac{U_2}{C_2 R_2}
\end{aligned}
```

$U_1$ represents the voltage across $R_1$ and $C_1$, while $U_2$ represents the voltage across $R_2$ and $C_2$. Furthermore, the functional connection of SOC may be used to estimate any parameter in the model. The SOC of the battery has been defined as an integral over current which we have already viewed.

The coulmb counting method is as follows:
```math
\text{SOC} = \text{SOC}_0 + \frac{1}{C_N} \int_0^t \eta\text{I}_{\text{batt}} \, dt
```

For convenience, η, representing Coulomb efficiency is set to 1 in the calculation. Moreover, the capacity of the battery is represented by $C_N$, where $\text{SOC}_0$ is the starting battery charge. In addition, for the stated dynamic 2-RC ECM, $SOC U_1 U_2$ has been selected as the state variable. The state space equation may be discretized using as follows.

```math
\begin{aligned}
\begin{bmatrix}
U_1(k+1) \\
U_2(k+1) \\
SOC(k+1)
\end{bmatrix}
&=
\begin{bmatrix}
e^{-\frac{\Delta t}{\zeta_1}} & 0 & 0 \\
0 & e^{-\frac{\Delta t}{\zeta_2}} & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
U_1(k) \\
U_2(k) \\
SOC(k)
\end{bmatrix}
+
\begin{bmatrix}
R_1\left(1 - e^{-\frac{\Delta t}{\zeta_1}}\right) \\
R_2\left(1 - e^{-\frac{\Delta t}{\zeta_2}}\right) \\
-\frac{\Delta t}{C_N}
\end{bmatrix}
I(k) + W(k)
\\
U_L(k) &= U_{oc}\left(SOC(k)\right) - R_0 I_L(k) - U_1(k) - U_2(k) + v_k
\end{aligned}
```

The observation and state equation of the system are represented above, where:
- Δt represents the sampling period.
- $\zeta_1 \, \zeta_2$  are the time constants where $\zeta_2=C_2 R_2$ and $\zeta_1 = C_1 R_1$ hold.
- $w_k$ and $v_k$ denote process and observation noise.
- $C_N$ denotes the rated capacity of the battery. 
- $k$ is the current time point and the future time point is represented by $k + 1$.

This can also be expressed as below:
```math
\begin{aligned}
U_0 &= I \cdot R_0 \\
\frac{U_1}{R_1} + C_{1} \frac{dU_1}{dt} &= I_{\text{batt}} \\
\frac{U_2}{R_2} + C_{2} \frac{dU_2}{dt} &= I_{\text{batt}} \\
\end{aligned}
```

### TODO
- [ ] 11min - Onboarding webinar watch

### Reference

1. https://www.sciencedirect.com/science/article/pii/S1452398124001159
2. https://www.sciencedirect.com/science/article/pii/S2352152X24018905
3. https://www.sciencedirect.com/science/article/pii/S2352152X23014317
4. https://ieeexplore.ieee.org/abstract/document/7458455