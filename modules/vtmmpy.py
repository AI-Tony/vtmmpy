import numpy as np

class TMM:
    def __init__(self, 
            freq, 
            theta, 
            f_scale=1e12, 
            l_scale=1e-9, 
            incident_medium="air", 
            transmitted_medium="air"): 

        self.__theta = theta * np.pi/180 
        self.__freq = freq * f_scale 
        self.__f_scale = f_scale
        self.__l_scale = l_scale
        self.__materialsProperties = {} 
        self.__incident_medium = incident_medium 
        self.__transmitted_medium = transmitted_medium 

    def reflection(self, polarization):
        try:
            if polarization == "TE":
                ni = 1 
                nt = 1 
            elif polarization == "TM":
                ni = self.__materialsProperties[self.__incident_medium][0] 
                nt = self.__materialsProperties[self.__transmitted_medium][0] 
            bi = self.__materialsProperties[self.__incident_medium][1] 
            bt = self.__materialsProperties[self.__transmitted_medium][1] 
            M  = self.__transferMatrix(polarization) 
            M  = self.__inverse(M) 
            r  = -(M[0,0,:,:,:]*bi*nt*nt - M[1,1,:,:,:]*bt*ni*ni + 1j*(M[1,0,:,:,:]*nt*nt*ni*ni + M[0,1,:,:,:]*bi*bt))/\
                  (M[0,0,:,:,:]*bi*nt*nt + M[1,1,:,:,:]*bt*ni*ni - 1j*(M[1,0,:,:,:]*nt*nt*ni*ni - M[0,1,:,:,:]*bi*bt)) 
            return r 
        except:
            raise Exception("no designs present. Use addDesign() method to add a design") 

    def transmission(self, polarization):
        try:
            if polarization == "TE":
                ni = 1 
                nt = 1 
            elif polarization == "TM":
                ni = self.__materialsProperties[self.__incident_medium][0] 
                nt = self.__materialsProperties[self.__transmitted_medium][0] 
            bi = self.__materialsProperties[self.__incident_medium][1] 
            bt = self.__materialsProperties[self.__transmitted_medium][1] 
            M  = self.__transferMatrix(polarization) 
            M  = self.__inverse(M) 
            t  = -2*ni*nt*bi / (M[0,0,:,:,:]*bi*nt*nt + M[1,1,:,:,:]*bt*ni*ni - 1j*(M[1,0,:,:,:]*nt*nt*ni*ni - M[0,1,:,:,:]*bi*bt)) 
            return t 
        except:
            raise Exception("no designs present. Use addDesign() method to add a design") 

    def addDesign(self, materials, thicknesses): 
        materials = np.array(materials) 
        thicknesses = np.array(thicknesses) * self.__l_scale 
        if len(materials) != len(thicknesses):
            raise ValueError("materials and thicknesses lists must be the same length (got {} and {})".format(len(materials), len(thicknesses))) 
        try:
            shape = self.__materials.shape
            shape = (shape[0]+1, shape[1]) 
            self.__materials = np.append(self.__materials, materials).reshape(shape) 
            self.__thicknesses = np.append(self.__thicknesses, thicknesses).reshape(shape) 
        except:
            self.__materials = materials[None,:] 
            self.__thicknesses = thicknesses[None,:] 
        self.__dims = ( len(self.__materials), len(self.__freq), len(self.__theta) ) 
        self.__layers = self.__materials.shape[1] 
        self.__calculateMaterialProperties() 

    def __calculateMaterialProperties(self): 
        w = 2*np.pi*self.__freq[:,None] * np.ones(self.__dims[1:]) 
        k0 = w / 3e8 
        kx = k0 * np.sin( self.__theta ) 
        material_set = set( self.__materials.flatten() )
        material_set.add( self.__incident_medium )
        material_set.add( self.__transmitted_medium ) 
        for mat in material_set:
            if mat not in self.__materialsProperties.keys(): 
                n  = self.__refractiveIindex(mat, w) 
                beta = np.sqrt( k0**2 * n**2 - kx**2 ) 
                self.__materialsProperties["{}".format(mat)] = (n, beta) 

    def __refractiveIindex(self, mat, omega):
        ri = np.ones( self.__dims[1:] ) 
        if mat=="air": return 1.0 * ri 
        elif mat=="sio2": return 1.45 * ri 
        elif mat=="tio2": return 2.45 * ri 
        elif mat=="sin": return 1.99 * ri
        elif mat=="ito": 
            w, gamma, E_inf, wp2 =  omega, 2.05e14, 3.91, 2.65e15**2 
            eps = E_inf - wp2 / ( w**2 + 1j*gamma*w )
            n = np.sqrt(  eps.real + np.sqrt( eps.real**2 + eps.imag**2 ) ) / np.sqrt(2) 
            k = np.sqrt( -eps.real + np.sqrt( eps.real**2 + eps.imag**2 ) ) / np.sqrt(2) 
            return ( n + 1j*k ) * ri

    def __transferMatrix(self, polarization):
        M            = np.zeros((2, 2, *self.__dims), dtype='cfloat') 
        M[0,0,:,:,:] = np.ones(self.__dims, dtype='cfloat') 
        M[1,1,:,:,:] = np.ones(self.__dims, dtype='cfloat') 
        for i in np.arange(self.__layers-1, -1, -1):
            m = self.__subMatrix( self.__materials[:,i], self.__thicknesses[:,i], polarization ) 
            M = self.__matmul(M, m) 
        return M 

    def __subMatrix(self, materials, thicknesses, polarization): 
        d = thicknesses[:,None,None] * np.ones(self.__dims) 
        n    = np.empty(self.__dims, dtype="cfloat") 
        beta = np.empty(self.__dims, dtype="cfloat") 
        for i, mat in enumerate(materials): 
            if polarization == "TE":
                n = 1
            elif polarization == "TM":
                n[i,:,:] = self.__materialsProperties[ mat ][0] 
            beta[i,:,:] = self.__materialsProperties[ mat ][1] 
        A =  np.cos( beta * d ) 
        B =  np.sin( beta * d ) * n * n / beta 
        C = -np.sin( beta * d ) * beta / (n * n) 
        D =  np.cos( beta * d )  
        return np.array( [ [A, B], [C, D] ], dtype='cfloat' ) 

    def __matmul(self, M, m):
        A = M[0,0,:,:,:]*m[0,0,:,:,:] + M[0,1,:,:,:]*m[1,0,:,:,:] 
        B = M[0,0,:,:,:]*m[0,1,:,:,:] + M[0,1,:,:,:]*m[1,1,:,:,:] 
        C = M[1,0,:,:,:]*m[0,0,:,:,:] + M[1,1,:,:,:]*m[1,0,:,:,:] 
        D = M[1,0,:,:,:]*m[0,1,:,:,:] + M[1,1,:,:,:]*m[1,1,:,:,:] 
        return np.array( [ [A, B], [C, D] ], dtype='cfloat' ) 

    def __inverse(self, M):
        try:
            det = M[0,0,:,:,:]*M[1,1,:,:,:] - M[0,1,:,:,:]*M[1,0,:,:,:] 
            A   = M[0,0,:,:,:] 
            B   = M[0,1,:,:,:] 
            C   = M[1,0,:,:,:] 
            D   = M[1,1,:,:,:] 
            return np.array( [ [D, -B], [-C, A] ], dtype='cfloat' ) / det 
        except ZeroDivisionError: 
            print("Warning: division by zero") 
            return np.array( [ [0, -0], [-0, 0] ], dtype='cfloat' ) 

    def summary(self):
        print("\n Summary")
        print(" --------------------------------------------") 
        print("  Number of designs:        ", len(self.__materials)) 
        print("  Frequency range (THz):    ", self.__freq[0]*1e-12, "-", self.__freq[-1]*1e-12) 
        print("  Angles of incidence:      ", np.rint(self.__theta[0]*180/np.pi), "-", np.rint(self.__theta[-1]*180/np.pi)) 
        print("  Incident medium:          ", self.__incident_medium) 
        print("  Transmitted medium:       ", self.__transmitted_medium) 
        print("") 

    def designs(self):
        try:
            zipped = zip(self.__materials,self.__thicknesses) 
            for pair in zipped:
                print(pair)
        except:
            raise Exception("add designs with addDesign() before trying to call designs() method")
