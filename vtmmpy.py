import numpy as np

class TMM:
    def __init__(self, 
            materials, 
            thicknesses, 
            freq, 
            theta, 
            f_scale=1e12, 
            l_scale=1e-9, 
            incident_medium="air", 
            transmitted_medium="air"):

        if materials.ndim == 1:
            self.__materials = materials[None,:] 
            self.__thicknesses = thicknesses[None,:] * l_scale 
        elif materials.ndim == 2:
            self.__materials = materials 
            self.__thicknesses = thicknesses * l_scale 
        else:
            raise ValueError("materials/thicknesses arrays have the wrong number of dimensions (should be 1 or 2)") 

        self.__layers = self.__materials.shape[1] 
        self.__theta = theta * np.pi/180 
        self.__freq = freq * f_scale 
        self.__materialsProperties = {} 
        self.__incident_medium = incident_medium 
        self.__transmitted_medium = transmitted_medium 
        self.__dims = ( len(self.__materials), len(freq), len(theta) ) 
        self.__calculateMaterialProperties() 

    def reflectionSpectra(self):
        bi = self.__materialsProperties[self.__incident_medium][1] 
        bt = self.__materialsProperties[self.__transmitted_medium][1] 
        M  = self.__transferMatrix() 
        M  = self.__inverse(M) 
        r  = -(M[0,0,:,:,:]*bi - M[1,1,:,:,:]*bt + 1j*(M[1,0,:,:,:] + M[0,1,:,:,:]*bi*bt))/\
              (M[0,0,:,:,:]*bi + M[1,1,:,:,:]*bt - 1j*(M[1,0,:,:,:] - M[0,1,:,:,:]*bi*bt)) 
        return r 

    def transmissionSpectra(self):
        bi = self.__materialsProperties[self.__incident_medium][1] 
        bt = self.__materialsProperties[self.__transmitted_medium][1] 
        M  = self.__transferMatrix() 
        M  = self.__inverse(M) 
        t  = -2*ni*nt*bi / (M[0,0,:,:,:]*bi + M[1,1,:,:,:]*bt - 1j*(M[1,0,:,:,:] - M[0,1,:,:,:]*bi*bt)) 
        return t 

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

    def __transferMatrix(self):
        M            = np.zeros((2, 2, *self.__dims), dtype='cfloat') 
        M[0,0,:,:,:] = np.ones(self.__dims, dtype='cfloat') 
        M[1,1,:,:,:] = np.ones(self.__dims, dtype='cfloat') 
        for i in np.arange(self.__layers-1, 0, -1):
            m = self.__subMatrix( self.__materials[:,i], self.__thicknesses[:,i] ) 
            M = self.__matmul(M, m) 
        return M 

    def __subMatrix(self, materials, thicknesses): 
        d = thicknesses[:,None,None] * np.ones(self.__dims) 
        n    = np.empty(self.__dims, dtype="cfloat") 
        beta = np.empty(self.__dims, dtype="cfloat") 
        for i, mat in enumerate(materials): 
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
