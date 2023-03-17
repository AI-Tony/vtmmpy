# **Vectorized Transfer Matrix Method Python** 
The transfer matrix method (TMM) is an analytic approach for obtaining the reflection and transmission coefficients in stratified media. vtmmpy is a vectorised implementation of the TMM written in Python. It has a focus on speed and ease of use. 

![](images/MTM.png)

### **Mathematical background**

---

$$
\mathbf{M}_{TM}^{(\ell)} = 
\begin{bmatrix}
 \cos(\beta_\ell d_\ell) & \sin(\beta_\ell d_\ell)n_\ell^2/\beta_\ell \\
-\sin(\beta_\ell d_\ell)\beta_\ell/n_\ell^2 & \cos(\beta_\ell d_\ell)
\end{bmatrix}
\hspace{1cm}
\mathbf{M}_{TE}^{(\ell)} = 
\begin{bmatrix}
 \cos(\beta_\ell d_\ell) & \sin(\beta_\ell d_\ell)/\beta_\ell \\
-\sin(\beta_\ell d_\ell)\beta_\ell & \cos(\beta_\ell d_\ell)
\end{bmatrix}
$$ 

$$
\mathbf{M} = \mathbf{M}^{(L)}\cdots\mathbf{M}^{(2)}\cdot\mathbf{M}^{(1)}
$$

$$
r_{TM} = -\frac{\mathbf{M}_{TM}^{-1}n_t^2\beta_i-\mathbf{M}_{TM}^{-1}n_i^2\beta_t+i[\mathbf{M}_{TM}^{-1}n_i^2n_t^2+\mathbf{M}_{TM}^{-1}\beta_i\beta_t]}{\mathbf{M}_{TM}^{-1}n_t^2\beta_i+\mathbf{M}_{TM}^{-1}n_i^2\beta_t-i[\mathbf{M}_{TM}^{-1}n_i^2n_t^2-\mathbf{M}_{TM}^{-1}\beta_i\beta_t]}
$$

$$
r_{TE} = -\frac{\mathbf{M}_{TM}^{-1}\beta_i-\mathbf{M}_{TM}^{-1}\beta_t+i[\mathbf{M}_{TM}^{-1}+\mathbf{M}_{TM}^{-1}\beta_i\beta_t]}{\mathbf{M}_{TM}^{-1}\beta_i+\mathbf{M}_{TM}^{-1}\beta_t-i[\mathbf{M}_{TM}^{-1}-\mathbf{M}_{TM}^{-1}\beta_i\beta_t]}
$$

$$
t_{TM} = -\frac{2n_in_t\beta_i}{\mathbf{M}_{TM}^{-1}n_t^2\beta_i+\mathbf{M}_{TM}^{-1}n_i^2\beta_t-i[\mathbf{M}_{TM}^{-1}n_i^2n_t^2-\mathbf{M}_{TM}^{-1}\beta_i\beta_t]}
$$

$$
t_{TE} = -\frac{2\beta_i}{\mathbf{M}_{TM}^{-1}\beta_i+\mathbf{M}_{TM}^{-1}\beta_t-i[\mathbf{M}_{TM}^{-1}-\mathbf{M}_{TM}^{-1}\beta_i\beta_t]}
$$

### **Installation**

---

Working on creating a pip package for the project. For now, simply run the commands below

```
git clone git@github.com:AI-Tony/vtmmpy.git
cd vtmmpy/modules
mv vtmmpy.py <YOUR PATH OR PROJECT DIRECTORY> 
```

### **Usage**

--- 

Import the vtmmpy module.

```
import vtmmpy
```

Create an instance of the ```TMM``` class. 

```
freq = np.linspace(170, 210, 30) 
theta = np.array(0, 60, 60) 

tmm = vtmmpy.TMM(freq, 
                theta, 
                f_scale=1e12, 
                l_scale=1e-9, 
                incident_medium="air", 
                transmitted_medium="air") 
```

- freq: a numpy array representing the spectral range of interest. 
- theta: a numpy array of one or more angles of incidence. 
- f_scale (optional): input frequency scale, default is terahertz.
- l_scale (optional): input length scale, default is nanometers.
- incident_medium (optional): incident medium, default is air.
- transmitted_medium (optional): transmitted medium, default is air. 

Add multilayer metamaterial designs with the ```addDesign()``` method.

```
materials = ["sio2", "tio2", "sio2", "tio2", "sio2"] 
thicknesses = [54, 92, 134, 112, 68] 

tmm.addDesign(materials, thicknesses)
```

- materials: list of materials 
- thicknesses: list of the corresponding material thicknesses 

Optionally call the ```summary()``` and/or ```designs()``` methods to view the data currently held by the instance.

```
tmm.summary() 
tmm.designs() 
```

Calculate the reflection/transmission coefficients by calling the appropriate method. You should specify wether you want the transverse magnetic/electric polarization by supplying the "TM" or "TE" flag, respectively.

```
RTM = tmm.reflection("TM") 
RTE = tmm.reflection("TE") 
TTM = tmm.transmission("TM") 
TTE = tmm.transmission("TE") 
```

Tips: 
 - The ```reflection()``` and ```transmission()``` methods return both complex parts. Use Python's built-in ```abs()``` function to obtain the magnitude.
 - The intensity is the square of the magnitude (eg. ```abs(reflection("TM"))**2```). 
 - The minimum number of dimensions for the output of ```reflection()``` and ```transmission()``` is 2. Therefore, when printing/plotting results, an index must be provided. 

### **Examples**

--- 

![](images/all.png)
