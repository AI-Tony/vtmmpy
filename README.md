# **Vectorized Transfer Matrix Method Python** 
The transfer matrix method (TMM) is an analytic approach for obtaining the reflection and transmission coefficients in stratified media. vtmmpy is a vectorised implementation of the TMM written in Python. It has a focus on speed and ease of use. 

![](https://github.com/AI-Tony/vtmmpy/blob/main/images/MTM.png?raw=true) 

### **Installation**

---

```
pip install vtmmpy 
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
materials = ["Ag", "SiO2", "Ag", "SiO2", "Ag"] 
thicknesses = [15, 85, 15, 85, 15] 

tmm.addDesign(materials, thicknesses)
```

- materials: list of materials 
- thicknesses: list of the corresponding material thicknesses 

Internally, vtmmpy uses the [regidx](https://gitlab.com/benvial/refidx) Python package to download refractive index data from [refractiveindex.info](https://refractiveindex.info/) for your choosen materials and spectral range. At this point, you will be presented with a few options corresponding to the data source ("Page" dropdown on refractiveindex.info). Study these carefully and refer to [refractiveindex.info](https://refractiveindex.info/) for more detailed information about how the data were obtained. Your choice here could greatly impact the accuracy of your results.

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
 - ```reflection()``` and ```transmission()``` return an ndarray with a minimum of 2 dimensions. The first dimension always corresponds to the number of designs. Therefore, when printing/plotting results, you must always index the first dimension (even if you only have 1 design). 

### **Examples**

--- 



![](https://github.com/AI-Tony/vtmmpy/blob/main/images/2dplots.png?raw=true)
