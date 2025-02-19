import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
from abc import ABC, abstractmethod
from sklearn.metrics import r2_score
from typing import List, Union
from sklearn.linear_model import LinearRegression

class basefit(ABC):
    
    def __init__(self):
        self.ydata = None
        self.epsilon = 1e-11
    
    @abstractmethod
    def fitFunction(self):
        pass
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def plots(self):
        pass
    
    def optimizationFunction(self,parameters):

        figureOfMerit = np.sum((self.ydata - self.fitFunction(parameters))**2);

        return figureOfMerit
    
    def eval(self, ytrue, ypred):
        return r2_score(ytrue, ypred)
        
    
class FitExpDecay(basefit):
    """Single exponential fit y = a0*exp(-x/a1)
    Args:
        xValues : x 
        yValues : y
        showplot : plot fitting results including intial estimation
        disp : display iterations during optimization
    
    Note : adapted from ultrafast relaxation matlab script
    """
    def __init__(self,
                 xValues : Union[List[float], np.ndarray], 
                 yValues : Union[List[float], np.ndarray], 
                 showplot : bool = False, 
                 disp : bool = False):
        
        super().__init__()
 
        self.result = None

        self.xdata = np.reshape(xValues,(1,len(xValues)));
        self.ydata = np.reshape(yValues,(1,len(yValues)));
        
        self.a = [0,0]

        # Lifetime & First moment
        self.a[1] = np.sum(self.xdata*np.abs(self.ydata))/np.sum(np.abs(self.ydata))

        # Amplitude
        smallestX = np.argmin(np.abs(self.xdata*(self.xdata>0)));
        #print(smallestX)
        self.a[0] = self.ydata[0,smallestX]*np.exp(self.xdata[0,smallestX]/self.a[1])
        #return para,newYData
        
        self.showplot = showplot
        self.disp = disp
        self.r2 = None


    def fitFunction(self,a):

        newyValues = a[0]*np.exp(-self.xdata/a[1]);

        return newyValues
    
    def fit(self):
        ## minimize
        para = minimize(self.optimizationFunction,self.a, method='Nelder-Mead',options={'disp': self.disp},tol=1e-6)
        #print(para)

        self.newYData = self.fitFunction(para.x)
        self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
        self.para = para
        
        if self.showplot : self.plots()
            
        self.result = {'a' : para.x[0],
                       'k' : 1/para.x[1]}
        return self
    
    def plots(self):
        plt.figure();
        plt.scatter( self.xdata.ravel(), self.ydata.ravel(),label ='raw',color='r')
        plt.plot( self.xdata.ravel(),self.fitFunction(self.a).ravel(),label ='initial',linestyle = '--',color='g')
        plt.plot( self.xdata.ravel(),self.newYData.ravel(),label='fit',linestyle = '-',color='b')
        plt.legend()
        plt.title('a: {}|k: {}|r2: {}'.format(round(self.para.x[0],3), round(1/self.para.x[1],3),round(self.r2,3)))

    def __repr__(self):
        return 'single exponential fit' 
    

class FitExpDecayOffset(basefit):
    """Single exponential + offset fit y = a0*exp(-x/a1) + a2
    Args:
        xValues : x 
        yValues : y
        showplot : plot fitting results including intial estimation
        disp : display iterations during optimization
    
    Note : adapted from ultrafast relaxation matlab script
    """

    def __init__(self,
                 xValues : Union[List[float], np.ndarray], 
                 yValues : Union[List[float], np.ndarray], 
                 showplot : bool = False, 
                 disp : bool = False):
        
        super().__init__()

        self.result = None

        self.xdata = np.reshape(xValues,(1,len(xValues)));
        self.ydata = np.reshape(yValues,(1,len(yValues)));
        
        self.a = [0,0,0]

        # Lifetime & First moment
        self.a[1] = np.sum(self.xdata*np.abs(self.ydata))/np.sum(np.abs(self.ydata))

        # Amplitude
        smallestX = np.argmin(np.abs(self.xdata*(self.xdata>0)));
        #print(smallestX)
        self.a[0] = self.ydata[0,smallestX]*np.exp(self.xdata[0,smallestX]/self.a[1])
        
        # offset
        last10 = int(0.1*self.xdata.shape[1]);
        self.a[2] = self.ydata[0,-last10:].mean()
        
        self.showplot = showplot
        self.disp = disp

    def fitFunction(self,a):

        newyValues = a[0]*np.exp(-self.xdata/a[1]) + a[2] ;

        return newyValues
    
    def fit(self):
        ## minimize
        para = minimize(self.optimizationFunction,self.a, method='Nelder-Mead',options={'disp': self.disp},tol=1e-6)
        #print(para)
        
        self.newYData = self.fitFunction(para.x)
        self.para = para
        self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
        
        if self.showplot : self.plots()
            
        self.result = {'a' : para.x[0],
                       'k' : 1/para.x[1],
                       'offset' : para.x[2]}
        
        return self
    
    def plots(self) :
        plt.figure();
        plt.scatter( self.xdata.ravel(), self.ydata.ravel(),label ='raw',color='r')
        plt.plot( self.xdata.ravel(),self.fitFunction(self.a).ravel(),label ='initial',linestyle = '--',color='g')
        plt.plot( self.xdata.ravel(),self.newYData.ravel(),label='fit',linestyle = '-',color='b')
        plt.legend();
        plt.title('a: {}|k: {}|off: {}|r2: {}'.format(round(self.para.x[0],3), 
                                                      round(1/self.para.x[1],3),
                                                      round(self.para.x[2],3),
                                                      round(self.r2,3)))

    
    def __repr__(self):
        return 'single exponential fit + offset' 
    
    
class FitDoubleExpDecay(basefit):
    """Double exponential + offset fit y = a0*exp(-x/a1) + a2*exp(-x/a3)
    Args:
        xValues : x 
        yValues : y
        showplot : plot fitting results including intial estimation
        disp : display iterations during optimization
    
    Note : adapted from ultrafast relaxation matlab script
    """

    def __init__(self,
                 xValues : Union[List[float], np.ndarray], 
                 yValues : Union[List[float], np.ndarray], 
                 showplot : bool = False, 
                 disp : bool = False):
     
        super().__init__()
        self.result = None

        self.xdata = np.reshape(xValues,(1,len(xValues)));
        self.ydata = np.reshape(yValues,(1,len(yValues)));
        
        self.a = [0,0,0,0]
        tt = [0,0]
        
        # Lifetime slow & First moment
        tt[1] = np.sum(self.xdata*np.abs(self.ydata))/np.sum(np.abs(self.ydata))

        # Amplitude slow
        smallestX = np.argmin(np.abs(self.xdata*(self.xdata>0)));
        #print(smallestX)
        tt[0] = self.ydata[0,smallestX]*np.exp(self.xdata[0,smallestX]/tt[1])
        
        # Improve Guess
        fac = 10;
        amp_slow = np.linspace(0,tt[0]+tt[0]/2,fac);
        t_slow = np.linspace(tt[1]-tt[1]/2,tt[1]+2*tt[1],fac);

        amp_fast = np.linspace(0,tt[0]+tt[0]/2,fac);
        t_fast = np.linspace(0,3/4*tt[1],fac);
        
        results = np.zeros((len(amp_slow),len(t_slow),len(amp_fast) ,len(t_fast )));

        optvals = 1e23
        #count = 0
        for id1, i1 in enumerate(amp_slow):
            for id2, i2 in enumerate(t_slow):
                for id3, i3 in enumerate(amp_fast):
                    for id4, i4 in enumerate(t_fast):
                        results[id1,id2,id3,id4] = self.optimizationFunction([i1,i2,i3,i4])
                        if results[id1,id2,id3,id4] < optvals:
                            best_ids = [id1,id2,id3,id4]
                            optvals = results[id1,id2,id3,id4]
                            #count +=1
        #I = np.argmin(results.ravel())
        self.a = [amp_slow[best_ids[0]],t_slow[best_ids[1]],amp_fast[best_ids[2]],t_fast[best_ids[3]]]
        for idx, k in enumerate(self.a):
            if k == 0:
                self.a[idx] = 1
                
        self.showplot = showplot
        self.disp = disp

    def fitFunction(self,a):
        
        newyValues = self.ExpFunction(a[:2]) + self.ExpFunction(a[2:]);

        return newyValues
    
    
    def ExpFunction(self,a):
        newyValues = a[0]*np.exp(-self.xdata/(a[1]+self.epsilon));
        return newyValues
    
    
    def fit(self):
        ## minimize
        para = minimize(self.optimizationFunction,self.a, method='Nelder-Mead',options={'disp': self.disp},tol=1e-6)
        
        self.newYData = self.fitFunction(para.x)
        self.para = para
        self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
        
        if self.showplot : self.plots()
        
        self.result = {'a_slow' : para.x[0],
                       'k_slow' : 1/para.x[1],
                       'a_fast' : para.x[2],
                       'k_fast' : 1/para.x[3]}
                    
        return self
    
    def plots(self) :
        plt.figure();
        plt.scatter( self.xdata.ravel(), self.ydata.ravel(),label ='raw',color='r')
        plt.plot( self.xdata.ravel(),self.fitFunction(self.a).ravel(),label ='initial',linestyle = '--',color='g')
        plt.plot( self.xdata.ravel(),self.newYData.ravel(),label='fit',linestyle = '-',color='b')
        plt.legend();

        plt.title('$a(slow)$: {} |$k(slow)$: {}|$a(fast)$: {}|$k(fast)$: {} |r2: {}'.format(round(self.para.x[0],2),
                                                                                            round(1/self.para.x[1],2),
                                                                                            round(self.para.x[2],2),
                                                                                            round(1/self.para.x[3],2),
                                                                                            round(self.r2,2)))
    
    def __repr__(self):
        return 'double exponential fit' 


class FitTanh(basefit):
    '''Tan hyperbolic fit y = a*(exp(2*x)-1)/(exp(2*x)+1)
       shifted x : x1 = (x - x0)* b
    Args:
        xValues : x 
        yValues : y
        showplot : plot fitting results including intial estimation
        disp : display iterations during optimization

    '''
    def __init__(self,
                 xValues : Union[List[float], np.ndarray], 
                 yValues : Union[List[float], np.ndarray], 
                 showplot : bool = False, 
                 disp : bool = False):
        
        super().__init__()
 
        self.result = None

        self.xdata = np.reshape(xValues,(1,len(xValues)));
        self.ydata = np.reshape(yValues,(1,len(yValues)));
        
        self.a = [0,0.8,0]
        
        # saturation
        self.a[2] = self.ydata.max()
        
        # center
        dummy_x = self.xdata[self.ydata>0][:3]
        dummy_y = self.ydata[self.ydata>0][:3]
        reg = LinearRegression().fit(dummy_x.reshape(-1,1), dummy_y)
        
        self.a[0] = -reg.intercept_/reg.coef_[0]
        
        print(self.a)
        self.showplot = showplot
        self.disp = disp
        self.r2 = None


    def fitFunction(self,a):
        
        dummy = np.exp(2*(self.xdata - a[0])*a[1])
        newyValues = a[2]*(dummy-1)/(dummy+1);

        return newyValues
    
    def fit(self):
        ## minimize
        para = minimize(self.optimizationFunction,self.a, method='Nelder-Mead',options={'disp': self.disp},tol=1e-6)
        #print(para)

        self.newYData = self.fitFunction(para.x)
        self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
        self.para = para
        
        if self.showplot : self.plots()
        
        #self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
            
        self.result = {'a' : para.x[2],
                       'x0' : para.x[0],
                       'b':para.x[1]}
        return self
    
    def plots(self):
        plt.figure();
        plt.scatter( self.xdata.ravel(), self.ydata.ravel(),label ='raw',color='r')
        plt.plot( self.xdata.ravel(),self.fitFunction(self.a).ravel(),label ='initial',linestyle = '--',color='g')
        plt.plot( self.xdata.ravel(),self.newYData.ravel(),label='fit',linestyle = '-',color='b')
        plt.legend()
        plt.title('a: {}|x0: {}|b: {}|r2: {}'.format(round(self.para.x[2],3), 
                                                     round(self.para.x[0],3),
                                                     round(self.para.x[1],3),
                                                     round(self.r2,3)))
        
    def predict(self, x):
        a = self.para.x
        dummy = np.exp(2*(x - a[0])*a[1])
        
        return a[2]*(dummy-1)/(dummy+1);

    def __repr__(self):
        return 'Tan-hyperbolic fit' 


class FitCubicPoly(basefit):
    ''' Third-degree polynomial fit y = a0x^3 + a1x^2 + a2x + a3
    Args:
        xValues : x 
        yValues : y
        showplot : plot fitting results including intial estimation
        disp : display iterations during optimization
    '''
    def __init__(self,
                 xValues : Union[List[float], np.ndarray], 
                 yValues : Union[List[float], np.ndarray], 
                 showplot : bool = False, 
                 disp : bool = False):
        
        super().__init__()
 
        self.result = None

        self.xdata = np.reshape(xValues,(1,len(xValues)));
        self.ydata = np.reshape(yValues,(1,len(yValues)));
        
        self.a = [0,0,0,0]

        # center
        dummy_x = self.xdata[self.ydata>0][:3]
        dummy_y = self.ydata[self.ydata>0][:3]
        reg = LinearRegression().fit(dummy_x.reshape(-1,1), dummy_y)
        
        print(self.a)
        self.showplot = showplot
        self.disp = disp
        self.r2 = None


    def fitFunction(self,a):
        newyValues =  a[0]*self.xdata**3 + a[1]*self.xdata**2 + a[2]*self.xdata + a[3];
        return newyValues
    
    def fit(self):
        ## Fit the data to the polynomial function using curve_fit
        para, _ = curve_fit(self.polynomial_func, self.xdata[0], self.ydata[0])
        # para = minimize(self.optimizationFunction,self.a, method='Nelder-Mead',options={'disp': self.disp},tol=1e-6)

        # self.newYData = self.fitFunction(para.x)
        # self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
        # self.para = para
        
        # if self.showplot : self.plots()
        
        # #self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())

        self.para = para
        
        # TODO Fix the actual values
        self.result = {'a' : para[2],
                       'x0' : para[0],
                       'b': para[1]}
        return self
    
    def plots(self):
        plt.figure();
        plt.scatter( self.xdata.ravel(), self.ydata.ravel(),label ='raw',color='r')
        plt.plot( self.xdata.ravel(),self.fitFunction(self.a).ravel(),label ='initial',linestyle = '--',color='g')
        plt.plot( self.xdata.ravel(),self.newYData.ravel(),label='fit',linestyle = '-',color='b')
        plt.legend()
        # TODO Fix the actual values
        plt.title('a: {}|x0: {}|b: {}|r2: {}'.format(round(self.para.x[2],3), 
                                                     round(self.para.x[0],3),
                                                     round(self.para.x[1],3),
                                                     round(self.r2,3)))
        
    def predict(self, x):
        a = self.para
        dummy = a[0]*x**3 + a[1]*x**2 + a[2]*x + a[3]
        return dummy

    def polynomial_func(self, x, a, b, c, d):
        return a* x**3 + b* x**2 + c*x + d
 
    def __repr__(self):
        return 'Cubic polynomial fit' 


class FitBiquadPoly(basefit):
    ''' Fourth-degree polynomial fit y = a0x^4 + a1x^3 + a2x^2 + a3x + a4
    Args:
        xValues : x 
        yValues : y
        showplot : plot fitting results including intial estimation
        disp : display iterations during optimization
    '''
    def __init__(self,
                 xValues : Union[List[float], np.ndarray], 
                 yValues : Union[List[float], np.ndarray], 
                 showplot : bool = False, 
                 disp : bool = False):
        
        super().__init__()
 
        self.result = None

        self.xdata = np.reshape(xValues,(1,len(xValues)));
        self.ydata = np.reshape(yValues,(1,len(yValues)));
        
        self.a = [0,0,0,0,0]
        
        print(self.a)
        self.showplot = showplot
        self.disp = disp
        self.r2 = None


    def fitFunction(self,a):
        newyValues =  a[0]*self.xdata**4 + a[1]*self.xdata**3 + a[2]*self.xdata**2 + a[3]*self.xdata + a[4]
        return newyValues
    
    def fit(self):
        ## Fit the data to the polynomial function using curve_fit
        para, _ = curve_fit(self.polynomial_func, self.xdata[0], self.ydata[0])
        # para = minimize(self.optimizationFunction,self.a, method='Nelder-Mead',options={'disp': self.disp},tol=1e-6)

        # self.newYData = self.fitFunction(para.x)
        # self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())
        # self.para = para
        
        # if self.showplot : self.plots()
        
        # #self.r2 = self.eval(self.ydata.ravel(),self.newYData.ravel())

        self.para = para
        
        # TODO Fix the actual values
        self.result = {'a' : para[2],
                       'x0' : para[0],
                       'b': para[1]}
        return self
    
    def plots(self):
        plt.figure();
        plt.scatter( self.xdata.ravel(), self.ydata.ravel(),label ='raw',color='r')
        plt.plot( self.xdata.ravel(),self.fitFunction(self.a).ravel(),label ='initial',linestyle = '--',color='g')
        plt.plot( self.xdata.ravel(),self.newYData.ravel(),label='fit',linestyle = '-',color='b')
        plt.legend()
        # TODO Fix the actual values
        plt.title('a: {}|x0: {}|b: {}|r2: {}'.format(round(self.para.x[2],3), 
                                                     round(self.para.x[0],3),
                                                     round(self.para.x[1],3),
                                                     round(self.r2,3)))
        
    def predict(self, x):
        a = self.para
        dummy = a[0]*x**4 + a[1]*x**3 + a[2]*x**2 + a[3]*x + a[4]
        return dummy

    def polynomial_func(self, x, a, b, c, d, e):
        return a* x**4 + b* x**3 + c*x**2 + d*x + e
 
    def __repr__(self):
        return 'Biquadratic polynomial fit' 
