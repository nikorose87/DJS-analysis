
#  Muscle contributions to support and progression over a range of walking speeds.
## Authors: Liu, May Q; Anderson, Frank C; Schwartz, Michael H; Delp, Scott L.

**Abstract**:_"Muscles actuate walking by providing vertical support and forward progression of the mass center. To quantify muscle contributions to vertical support and forward progression (i.e., vertical and fore-aft accelerations of the mass center) over a range of walking speeds, three-dimensional muscle-actuated simulations of gait were generated and analyzed for eight subjects walking overground at very slow, slow, free, and fast speeds. We found that gluteus maximus, gluteus medius, vasti, hamstrings, gastrocnemius, and soleus were the primary contributors to support and progression at all speeds. With the exception of gluteus medius, contributions from these muscles generally increased with walking speed. During very slow and slow walking speeds, vertical support in early stance was primarily provided by a straighter limb, such that skeletal alignment, rather than muscles, provided resistance to gravity. When walking speed increased from slow to free, contributions to support from vasti and soleus increased dramatically. Greater stance-phase knee flexion during free and fast walking speeds caused increased vasti force, which provided support but also slowed progression, while contralateral soleus simultaneously provided increased propulsion. This study provides reference data for muscle contributions to support and progression over a wide range of walking speeds and highlights the importance of walking speed when evaluating muscle function."_

**Purpose**: The purpose with this paper was to take the simulation data of all subjects in gait, generated from RRA, Inverse Kinematics (IK) and Inverse Dynamics (ID), in order to generate the DJS of the ankle joint.


## Data method acquisition
_"These data were used to generate subject-specific simulations at each walking speed (Fig. 1).We calculated muscle contributions to support and progression with a perturbation analysis (Liu et al., 2006). A repeated measures analysis of variance identified the effects of walking speed on muscle contributions to mass center accelerations."_
<center><img src="files/figures/speedsLiu1.png" alt="Drawing" style="width: 1000px;"/></center>
**Fig. 1** Musculoskeletal model used to generate three-dimensional simulations of walking for eight subjects, each walking at four speeds (very slow, slow, free, and fast)._ Taken from (Liu et al, 2008)

Walking speeds were categorized by (Hof,1996) and they used Opensim to adapt all subjects to model 2392. They asked to eight (8) subjects of different anthropometries as It is shown below:
![Table subjects used in this paper](files/figures/tablesubjects.png)
a = Speeds are reported in m/s and nondimensional units (actual speed normalized by $\sqrt(gL_{leg})$.

**Table 1** Experimental subject charactheristics. Taken from (Liu et al, 2008).

The process to obtain biomechanical rigid models was got throught scaling, followed by Inverse Kinematic process. After that, an Inverse Dynamic were applied to each model, and finally they applied external forces and moments (i.e., residuals) to the pelvis segment to compensate for dynamic inconsistencies between the measured
kinematics and the measured ground reaction forces (Kuo, 1998) to run RRA algorithm. 


```python
#Loading Libraries to develop the algorithm
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
from bokeh.plotting import figure
from bokeh.io import show, reset_output, output_notebook
import warnings
import numpy as np
from bokeh.layouts import row, column, gridplot
from bokeh.palettes import Paired12
from scipy.integrate import simps
from sklearn.svm    import SVR
from sklearn import linear_model
warnings.filterwarnings('ignore')
output_notebook()
#InteractiveShell.ast_node_interactivity = "last"
```



    <div class="bk-root">
        <a href="https://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="17ace701-dea2-4203-8e24-d634022c0a72">Loading BokehJS ...</span>
    </div>





```python
# Locating all moments dirs and anthropometric info of each subject
Dir = "/home/nikorose/Dropbox/Gait_Analysis_data/Downloaded/Liu/"
subjectsnames = ["GIL01","GIL02","GIL03","GIL04","GIL06","GIL08","GIL11", "GIL12"]
gender = ["F","F","M","F","F","F","F","M"]
age = [10.2, 14.6, 13.8, 11.3, 14.1, 14.5, 18.0, 7]
mass = [41.1, 66.0, 41.6, 32.4, 81.9, 61.9, 63.1, 26.1]
leg_lenght = [0.77, 0.9, 0.84, 0.72, 0.81, 0.94, 0.84, 0.66]
speeds = ['fast', 'free', 'slow', 'xslow'] 
measure_info = pd.DataFrame([subjectsnames, gender, age, mass, leg_lenght], index = ['Simulation ID', 'gender', 'Age (years)', 'mass (kg)','Leg lenght (m)'])
#print ("Anthropometric information of subjects")
measure_info



```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Simulation ID</th>
      <td>GIL01</td>
      <td>GIL02</td>
      <td>GIL03</td>
      <td>GIL04</td>
      <td>GIL06</td>
      <td>GIL08</td>
      <td>GIL11</td>
      <td>GIL12</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>F</td>
      <td>F</td>
      <td>M</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>F</td>
      <td>M</td>
    </tr>
    <tr>
      <th>Age (years)</th>
      <td>10.2</td>
      <td>14.6</td>
      <td>13.8</td>
      <td>11.3</td>
      <td>14.1</td>
      <td>14.5</td>
      <td>18</td>
      <td>7</td>
    </tr>
    <tr>
      <th>mass (kg)</th>
      <td>41.1</td>
      <td>66</td>
      <td>41.6</td>
      <td>32.4</td>
      <td>81.9</td>
      <td>61.9</td>
      <td>63.1</td>
      <td>26.1</td>
    </tr>
    <tr>
      <th>Leg lenght (m)</th>
      <td>0.77</td>
      <td>0.9</td>
      <td>0.84</td>
      <td>0.72</td>
      <td>0.81</td>
      <td>0.94</td>
      <td>0.84</td>
      <td>0.66</td>
    </tr>
  </tbody>
</table>
</div>



**Table 2** Anthropometric features of subjects.


```python
Temporal_params = pd.read_excel(Dir+"TableLiuIDs.xlsx")
Temporal_params
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Simulation ID</th>
      <th>Speed</th>
      <th>Stance limb</th>
      <th>Initial Contact (IC)</th>
      <th>Foot-flat (FF)</th>
      <th>Heel Off (HO)</th>
      <th>Next Initial contact</th>
      <th>swing HO</th>
      <th>Swing TO</th>
      <th>Swing Next IC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GIL01</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.83</td>
      <td>0.94</td>
      <td>1.56</td>
      <td>2.52</td>
      <td>0.83</td>
      <td>1.06</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GIL01</td>
      <td>slow</td>
      <td>R</td>
      <td>0.84</td>
      <td>0.96</td>
      <td>1.52</td>
      <td>2.42</td>
      <td>0.84</td>
      <td>1.07</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GIL01</td>
      <td>free</td>
      <td>R</td>
      <td>0.59</td>
      <td>0.68</td>
      <td>1.08</td>
      <td>1.73</td>
      <td>0.48</td>
      <td>0.74</td>
      <td>1.18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GIL01</td>
      <td>fast</td>
      <td>L</td>
      <td>0.49</td>
      <td>0.56</td>
      <td>0.84</td>
      <td>1.42</td>
      <td>0.36</td>
      <td>0.59</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>4</th>
      <td>GIL02</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.98</td>
      <td>1.08</td>
      <td>3.00</td>
      <td>2.94</td>
      <td>0.98</td>
      <td>1.36</td>
      <td>1.96</td>
    </tr>
    <tr>
      <th>5</th>
      <td>GIL02</td>
      <td>slow</td>
      <td>L</td>
      <td>0.66</td>
      <td>0.73</td>
      <td>1.29</td>
      <td>2.06</td>
      <td>0.63</td>
      <td>0.88</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GIL02</td>
      <td>free</td>
      <td>L</td>
      <td>0.56</td>
      <td>0.64</td>
      <td>0.98</td>
      <td>1.65</td>
      <td>0.48</td>
      <td>0.68</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>7</th>
      <td>GIL02</td>
      <td>fast</td>
      <td>L</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>0.73</td>
      <td>1.45</td>
      <td>0.32</td>
      <td>0.57</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GIL03</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.77</td>
      <td>0.92</td>
      <td>1.37</td>
      <td>2.39</td>
      <td>0.65</td>
      <td>1.01</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GIL03</td>
      <td>slow</td>
      <td>L</td>
      <td>0.76</td>
      <td>0.88</td>
      <td>1.40</td>
      <td>2.23</td>
      <td>0.73</td>
      <td>0.96</td>
      <td>1.52</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GIL03</td>
      <td>free</td>
      <td>L</td>
      <td>0.51</td>
      <td>0.60</td>
      <td>0.93</td>
      <td>1.51</td>
      <td>0.44</td>
      <td>0.62</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GIL03</td>
      <td>fast</td>
      <td>R</td>
      <td>0.36</td>
      <td>0.41</td>
      <td>0.56</td>
      <td>1.06</td>
      <td>0.21</td>
      <td>0.40</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>12</th>
      <td>GIL04</td>
      <td>very slow</td>
      <td>L</td>
      <td>0.94</td>
      <td>1.24</td>
      <td>1.74</td>
      <td>2.93</td>
      <td>0.84</td>
      <td>1.28</td>
      <td>1.95</td>
    </tr>
    <tr>
      <th>13</th>
      <td>GIL04</td>
      <td>slow</td>
      <td>L</td>
      <td>0.60</td>
      <td>0.71</td>
      <td>1.11</td>
      <td>1.83</td>
      <td>0.45</td>
      <td>0.74</td>
      <td>1.25</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GIL04</td>
      <td>free</td>
      <td>R</td>
      <td>0.57</td>
      <td>0.66</td>
      <td>1.00</td>
      <td>1.69</td>
      <td>0.42</td>
      <td>0.67</td>
      <td>1.11</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GIL04</td>
      <td>fast</td>
      <td>L</td>
      <td>0.51</td>
      <td>0.60</td>
      <td>0.95</td>
      <td>1.59</td>
      <td>0.43</td>
      <td>0.61</td>
      <td>1.06</td>
    </tr>
    <tr>
      <th>16</th>
      <td>GIL06</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.97</td>
      <td>1.14</td>
      <td>1.84</td>
      <td>3.06</td>
      <td>0.96</td>
      <td>1.33</td>
      <td>1.97</td>
    </tr>
    <tr>
      <th>17</th>
      <td>GIL06</td>
      <td>slow</td>
      <td>R</td>
      <td>0.66</td>
      <td>0.74</td>
      <td>1.24</td>
      <td>1.99</td>
      <td>0.56</td>
      <td>0.84</td>
      <td>1.35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GIL06</td>
      <td>free</td>
      <td>R</td>
      <td>0.54</td>
      <td>0.63</td>
      <td>1.00</td>
      <td>1.67</td>
      <td>0.48</td>
      <td>0.67</td>
      <td>1.15</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GIL06</td>
      <td>fast</td>
      <td>R</td>
      <td>0.51</td>
      <td>0.59</td>
      <td>0.92</td>
      <td>1.55</td>
      <td>0.38</td>
      <td>0.60</td>
      <td>1.05</td>
    </tr>
    <tr>
      <th>20</th>
      <td>GIL08</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.85</td>
      <td>0.96</td>
      <td>1.61</td>
      <td>2.58</td>
      <td>0.80</td>
      <td>1.11</td>
      <td>1.76</td>
    </tr>
    <tr>
      <th>21</th>
      <td>GIL08</td>
      <td>slow</td>
      <td>R</td>
      <td>0.76</td>
      <td>0.87</td>
      <td>1.44</td>
      <td>2.23</td>
      <td>0.71</td>
      <td>0.98</td>
      <td>1.50</td>
    </tr>
    <tr>
      <th>22</th>
      <td>GIL08</td>
      <td>free</td>
      <td>R</td>
      <td>0.59</td>
      <td>0.68</td>
      <td>1.10</td>
      <td>1.76</td>
      <td>0.52</td>
      <td>0.72</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>23</th>
      <td>GIL08</td>
      <td>fast</td>
      <td>R</td>
      <td>0.48</td>
      <td>0.56</td>
      <td>0.77</td>
      <td>1.43</td>
      <td>0.38</td>
      <td>0.59</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>24</th>
      <td>GIL11</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.78</td>
      <td>0.92</td>
      <td>1.61</td>
      <td>2.56</td>
      <td>0.77</td>
      <td>1.11</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>25</th>
      <td>GIL11</td>
      <td>slow</td>
      <td>R</td>
      <td>0.78</td>
      <td>0.88</td>
      <td>1.36</td>
      <td>2.21</td>
      <td>0.68</td>
      <td>0.96</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>26</th>
      <td>GIL11</td>
      <td>free</td>
      <td>R</td>
      <td>0.54</td>
      <td>0.61</td>
      <td>0.96</td>
      <td>1.59</td>
      <td>0.46</td>
      <td>0.66</td>
      <td>1.09</td>
    </tr>
    <tr>
      <th>27</th>
      <td>GIL11</td>
      <td>fast</td>
      <td>L</td>
      <td>0.45</td>
      <td>0.52</td>
      <td>0.74</td>
      <td>1.34</td>
      <td>0.32</td>
      <td>0.54</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>28</th>
      <td>GIL12</td>
      <td>very slow</td>
      <td>R</td>
      <td>0.91</td>
      <td>0.97</td>
      <td>1.58</td>
      <td>2.77</td>
      <td>0.80</td>
      <td>1.19</td>
      <td>1.86</td>
    </tr>
    <tr>
      <th>29</th>
      <td>GIL12</td>
      <td>slow</td>
      <td>L</td>
      <td>0.82</td>
      <td>0.94</td>
      <td>1.52</td>
      <td>2.34</td>
      <td>0.81</td>
      <td>1.03</td>
      <td>1.66</td>
    </tr>
    <tr>
      <th>30</th>
      <td>GIL12</td>
      <td>free</td>
      <td>R</td>
      <td>0.45</td>
      <td>0.53</td>
      <td>0.84</td>
      <td>1.41</td>
      <td>0.39</td>
      <td>0.55</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>31</th>
      <td>GIL12</td>
      <td>fast</td>
      <td>L</td>
      <td>0.41</td>
      <td>0.48</td>
      <td>0.66</td>
      <td>1.29</td>
      <td>0.35</td>
      <td>0.48</td>
      <td>0.87</td>
    </tr>
  </tbody>
</table>
</div>



**Table 3** biomechanical parameters of subjects.

Given all parameters of each subject. We will proceed with the acquisition and refinement of data obtained from biomechanical simulations. First of all, We will plot the Inverse Kinematics of each subject and compared with RRA.


```python
cuadratura = 400
count = 0
shows = []
for i in subjectsnames:
    color = 0
    f = figure(x_axis_label='Time (s)', y_axis_label='Right Ankle Angle (Deg)', plot_width=cuadratura,
               plot_height=cuadratura, title="Ankle kinematics at stance phase from Liu Paper - Model "+i)
    for j in speeds:
        color += 1
        k = pd.read_table(Dir+'IK/'+i+"_"+j+"_ik.mot", header='infer', sep= r"\s*", engine='python', skiprows= 11, decimal=".")
        k1 = pd.read_table(Dir+'RRA/IK-RRA/'+i+"_"+j+"_RRA_Kinematics_q.mot", header='infer', sep= r"\s*", engine='python', skiprows= 15, decimal=".")
        f.line(x = k['time'],y= k['ankle_angle_r'], color = Paired12[color], line_width=3, legend = 'Quasi-stiffness model'+i+" "+j)
        f.legend.location = "bottom_right"
        minline = k.loc[k['ankle_angle_r'].idxmin()]
        f.ray(x=Temporal_params["Initial Contact (IC)"][count],y=minline['ankle_angle_r'], length=10, angle=1.57079633, color=Paired12[color],line_dash="4 4")
        f.ray(x=Temporal_params["Heel Off (HO)"][count], y=minline['ankle_angle_r'], length=10, angle=1.57079633,
              color=Paired12[color], line_dash="4 4")
        color += 1
        f.line(x = k1['time'],y= k1['ankle_angle_r'], color = Paired12[color], line_width=3)
        count += 1
    shows.append(f)
grid = gridplot([shows[0],shows[1]],[shows[2],shows[3]],[shows[4],shows[5]],[shows[6],shows[7]])
show(grid)
```



<div class="bk-root">
    <div class="bk-plotdiv" id="4ef9539f-e2a9-4e28-b7ce-84b3f478da67"></div>
</div>




**Fig. 2** Stance phase at different walking speeds for subjects of Liu models.

In the graphs showed above, We can appreciate that the partial kinematics of this study corresponds with RRA analysis done in Opensim Software. However, there is only stance phase walking information, we cannot obtain complete quasi-stiffness unless we predict rest of cycle.
Now, we are going to compare the Inverse Dynamics with RRA forces to obtain quasi-stiffness.


```python
# Plotting ID with RRA analysis
count = 0
shows = []
for i in subjectsnames:
    color = 0
    q = figure(x_axis_label='Right Ankle Angle (Deg)', y_axis_label="Moment (Nm)", plot_width=cuadratura,
               plot_height=cuadratura, title="Ankle quasi-stiffness at stance phase from Liu Paper - Model "+i)
    for j in speeds:
        color += 1
        k = pd.read_table(Dir + 'IK/' + i + "_" + j + "_ik.mot", header='infer', sep=r"\s*", engine='python',
                          skiprows=11, decimal=".")
        k2 = pd.read_table(Dir+'ID/'+i+"_"+j+"_ik_invdyn_force.sto", header='infer', sep= r"\s*", engine='python', skiprows= 6, decimal=".")
        q.circle(x = k['ankle_angle_r'],y= -k2['ankle_angle_r']/mass[color], color = Paired12[color], size=k['time']*3, legend = 'Quasi-stiffness model'+i+" "+j)
        q.legend.location = "top_left"
        minline = k.loc[k['ankle_angle_r'].idxmin()]
        count += 1
    shows.append(q)
grid = gridplot([shows[0],shows[1]],[shows[2],shows[3]],[shows[4],shows[5]],[shows[6],shows[7]])
show(grid)

```



<div class="bk-root">
    <div class="bk-plotdiv" id="4beaa273-4b64-4e56-8b9e-8d8330305a4e"></div>
</div>




**Fig. 3** Ankle Quasi-Stiffness at different walking speeds for subjects of Liu models.

Unfortunately, The results of forces with RRA predictions does not match with Inverse Dynamics, thus there are accuracy problems that RRA tool cannot do well.

As a matter of fact, the data obtained by the study is incomplete, as It was shown in the figures above, most of the quasi-stiffness slopes are not closed. Thus, total work of all subjects cannot be calculated. A possibly solution could be obtaining final predictions of each slope with the technique of Machine Learning. 

Other possibility for finishing the quasi-stiffness of each model is applying Forward Dynamics, which is a tool of Opensim project. To do these predictions of each model is necessary to apply Inverse Kinematics, RRA, CMC and therefore Forward Dynamics.
