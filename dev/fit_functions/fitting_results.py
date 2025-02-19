import string
import xlsxwriter
import os
import json
import numpy as np
from fit_functions.fittingmodels import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle

class readkinetics():
    def __init__(self, 
                path, 
                frame_interval = 1, #seconds
                fit_mode = 'all', 
                showplot = False, 
                skip_frame = 1 # skip frames starts from 0
                ):
        
        if not(os.path.exists(os.path.join(path,'data.json'))) : raise Exception('file not found!')
        # open file
        f = open(os.path.join(path, 'data.json'))
        self.data = json.load(f)
        f.close()
        self.path = path
        self.frame_interval = frame_interval/60 # convert to minute
        self.fit_mode = fit_mode
        self.model1 = None
        self.model2 = None
        self.model3 = None
        self.showplot = showplot
        self.skip_frame = skip_frame
        
        
    def fit(self):
        if self.fit_mode =='all' :
            print('single exponential fit' + '#'*100)
            self.model1 = readkinetics.fitexp1(self.data, self.path, 'result_exp1.xlsx', self.frame_interval, self.skip_frame)
            print('single exponential + offset fit' + '#'*100)
            self.model2 = readkinetics.fitexp1_offset(self.data, self.path, 'result_exp1_offset.xlsx', self.frame_interval,self.skip_frame)
            print('double exponential fit' + '#'*100)
            self.model3 = readkinetics.fitexp2(self.data, self.path, 'result_exp2.xlsx', self.frame_interval, self.skip_frame)
        elif self.fit_mode == 'exp1' :
            self.model1 = readkinetics.fitexp1(self.data, self.path, 'result_exp1.xlsx', self.frame_interval, self.showplot,self.skip_frame)
            
        elif self.fit_mode == 'exp1_offset' :
            self.model2 = readkinetics.fitexp1_offset(self.data, self.path, 'result_exp1_offset.xlsx', self.frame_interval, self.skip_frame)
            
        elif self.fit_mode == 'exp2' :
            self.model3 = readkinetics.fitexp2(self.data, self.path, 'result_exp2.xlsx', self.frame_interval,self.skip_frame)
            
        else : 
            self.model1 = readkinetics.fitexp1(self.data, self.path, 'result_exp1.xlsx', self.frame_interval, self.skip_frame)
        return self
        
        
    @staticmethod
    def fitexp1(data, path, fname = 'results.xlsx', frame_interval = 1, showplot = True, skip_frame = 1):
    
        workbook = xlsxwriter.Workbook(os.path.join(path,fname))

        fall = []
        row = 1
        for idx, (el,val) in enumerate(data.items()):
            fc = []
            sheet_data = workbook.add_worksheet('sample_'+str(idx+1))
            sheet_fits = workbook.add_worksheet('result_'+str(idx+1))
            
            plt.figure()
            for col, y in enumerate(val) : 

                x = np.arange(0,len(y),1) * frame_interval
                if skip_frame :
                    y = y[skip_frame :]
                    x = x[: -skip_frame]

                fc.append(FitExpDecay(x,y,showplot=False, disp = False).fit())

                sheet_data.write(string.ascii_letters[2*col+1].upper()+'1','well_raw_'+str(col+1))
                sheet_data.write_column(row, 2*col+1, y)

                sheet_data.write(string.ascii_letters[2*col+2].upper()+'1','well_fit_'+str(col+1))
                sheet_data.write_column(row, 2*col+2, fc[col].newYData.ravel().tolist())
                #print(fc[col].result)
                if showplot :
                    plt.plot(x,y)
                    plt.plot(x, fc[col].newYData.ravel())
                    plt.title('exp1_'+str(idx+1)+'|a:'+str(round(fc[col].result['a'],3)) + '|k'+str(round(fc[col].result['k'],3)))
                plt.xlabel('t[min]')
                plt.ylabel('sig.')
            plt.savefig(os.path.join(path,'fit_kinetics_exp1_col_'+str(idx)+'.jpg'),bbox_inches='tight')
                
                    
            sheet_data.write_column(row, 0 , x.tolist())
            sheet_data.write('A1','t[min]')

            sheet_fits.write('A1','a')
            sheet_fits.write('B1','k')
            sheet_fits.write('C1','r2')
            sheet_fits.write_column(row,0,[el.result['a'] for el in fc])
            sheet_fits.write_column(row,1,[el.result['k'] for el in fc])
            sheet_fits.write_column(row,2,[el.r2 for el in fc])

            # save xlsx
            fall.append(fc)

        workbook.close()

        return fall
        
        
    @staticmethod
    def fitexp1_offset(data, path, fname = 'results.xlsx', frame_interval = 1, showplot = True, skip_frame = 1):
        workbook = xlsxwriter.Workbook(os.path.join(path,fname))

        fall = []
        row = 1
        for idx, (el,val) in enumerate(data.items()):
            fc = []
            sheet_data = workbook.add_worksheet('sample_'+str(idx+1))
            sheet_fits = workbook.add_worksheet('result_'+str(idx+1))
            
            plt.figure()
            for col, y in enumerate(val) : 
                x = np.arange(0,len(y),1) * frame_interval
                if skip_frame :
                    y = y[skip_frame :]
                    x = x[: -skip_frame]
                    
                
                
                fc.append(FitExpDecayOffset(x,y,showplot=False, disp = False).fit())
                
                #sheet_data.write(string.ascii_letters[col+1].upper()+'1','well_'+str(col+1))
                #sheet_data.write_column(row, col+1, y)
                sheet_data.write(string.ascii_letters[2*col+1].upper()+'1','well_raw_'+str(col+1))
                sheet_data.write_column(row, 2*col+1, y)
                
                sheet_data.write(string.ascii_letters[2*col+2].upper()+'1','well_fit_'+str(col+1))
                sheet_data.write_column(row, 2*col+2, fc[col].newYData.ravel().tolist())
                
                if showplot :
                    plt.plot(x,y)
                    plt.plot(x, fc[col].newYData.ravel())
                    plt.title('exp1_off_'+str(idx+1)+'|a:'+str(round(fc[col].result['a'],3)) + '|k'+str(round(fc[col].result['k'],3))+'|offset'+str(round(fc[col].result['offset'],3)))
                plt.xlabel('t[min]')
                plt.ylabel('sig.')
                plt.figure(figsize= (6,4))
            plt.savefig(os.path.join(path,'fit_kinetics_exp1+offset_col_'+str(idx)+'.jpg'),bbox_inches='tight')
            
            sheet_data.write_column(row, 0 , x.tolist())
            sheet_data.write('A1','t[min]')
            
            sheet_fits.write('A1','a')
            sheet_fits.write('B1','k')
            sheet_fits.write('C1','offset')
            sheet_fits.write('D1','r2')
            sheet_fits.write_column(row,0,[el.result['a'] for el in fc])
            sheet_fits.write_column(row,1,[el.result['k'] for el in fc])
            sheet_fits.write_column(row,2,[el.result['offset'] for el in fc])
            sheet_fits.write_column(row,3,[el.r2 for el in fc])
            
            # save xlsx
            fall.append(fc)
            
        workbook.close()
        
        
        return fall
        
        
    @staticmethod
    def fitexp2(data, path, fname = 'results.xlsx', frame_interval = 1, showplot = True, skip_frame = 1):

        workbook = xlsxwriter.Workbook(os.path.join(path,fname))

        fall = []
        row = 1
        for idx, (el,val) in enumerate(data.items()):
            fc = []
            sheet_data = workbook.add_worksheet('sample_'+str(idx+1))
            sheet_fits = workbook.add_worksheet('result_'+str(idx+1))
            
            plt.figure()
            for col, y in enumerate(val) : 
                x = np.arange(0,len(y),1) * frame_interval
                if skip_frame :
                    y = y[skip_frame :]
                    x = x[: -skip_frame]
                
                fc.append(FitDoubleExpDecay(x,y,showplot=False, disp = False).fit())
                
                #sheet_data.write(string.ascii_letters[col+1].upper()+'1','well_'+str(col+1))
                #sheet_data.write_column(row, col+1, y)
                
                sheet_data.write(string.ascii_letters[2*col+1].upper()+'1','well_raw_'+str(col+1))
                sheet_data.write_column(row, 2*col+1, y)
                
                sheet_data.write(string.ascii_letters[2*col+2].upper()+'1','well_fit_'+str(col+1))
                sheet_data.write_column(row, 2*col+2, fc[col].newYData.ravel().tolist())
                
                if showplot :
                    plt.plot(x,y)
                    plt.plot(x, fc[col].newYData.ravel())
                    plt.title('exp2_'+str(idx+1)+'|a_slow:'+str(round(fc[col].result['a_slow'],3)) + '|k_slow'+str(round(fc[col].result['k_slow'],3))+\
                    '|a_fast:'+str(round(fc[col].result['a_fast'],3))+'|k_fast:'+str(round(fc[col].result['k_fast'],3)))
                plt.xlabel('t[min]')
                plt.ylabel('sig.')
                plt.figure(figsize=(6,4))
                
            plt.savefig(os.path.join(path,'fit_kinetics_exp2_col_'+str(idx)+'.jpg'),bbox_inches='tight')
                
            
            sheet_data.write_column(row, 0 , x.tolist())
            sheet_data.write('A1','t[min]')
            
            sheet_fits.write('A1','a_slow')
            sheet_fits.write('B1','k_slow')
            sheet_fits.write('C1','a_fast')
            sheet_fits.write('D1','k_fast')
            sheet_fits.write('E1','r2')
            
            sheet_fits.write_column(row,0,[el.result['a_slow'] for el in fc])
            sheet_fits.write_column(row,1,[el.result['k_slow'] for el in fc])
            sheet_fits.write_column(row,2,[el.result['a_fast'] for el in fc])
            sheet_fits.write_column(row,3,[el.result['k_fast'] for el in fc])
            sheet_fits.write_column(row,4,[el.r2 for el in fc])
            
            # save xlsx
            fall.append(fc)
            
        workbook.close()
        
        
        return fall
    

class readtitration():
    
    def __init__(self, path, params, larger_than_boundary: bool, linear_first:bool ,boundary = 0, showplot = True):
        
        if os.path.exists(os.path.join(path, 'titration')) :
            self.path = os.path.join(path, 'titration')
        else:
            raise FileNotFoundError('titration folder does not exist!')    
        self.sub_folders = os.listdir(self.path)
        self.files = [os.path.join(self.path, el,'titration_result.csv') for el in self.sub_folders]
        self.x = None
        self.y = None
        self.showplot = showplot
        self.params = params
        self.larger_than_boundary = larger_than_boundary
        self.boundary = boundary
        self.success = False
        self.linear_first = linear_first

    # used to skip some points when decay  
    def fit(self, fitting_function):
        for sub_folder, file in zip(self.sub_folders, self.files):
            print('running folder : '+sub_folder)
            df = pd.read_csv(file)
            self.x = df['vol'].to_numpy()
            # to work with the old data files
            try:
                self.y = df['mean'+'_'+self.params].to_numpy()
                self.y_error =df['std'+'_'+self.params].to_numpy()
            except:
                self.y = df['mean'].to_numpy()
                self.y_error = df['std'].to_numpy()
            self.save_path = os.path.join(self.path,sub_folder)
            self.__fit_titration_curve(fitting_function)
            self.__save_pickle()
            if self.showplot and self.success: 
                self.plots(fitting_function)
        return self    


    # used to select fitting function (tanh, biquad, or cubic)
    def __fit_titration_curve(self, fitting_function = "biquad"):
        x = self.x
        y = self.y
        y_error = self.y_error
        self.success = False
        num = 10000
        xq = np.linspace(x[0],x[-1],num)
        print(y)
        # splitting the graph for fitting
        y1 = np.empty(0)
        y2 = np.empty(0)
        x1 = np.empty(0)
        x2 = np.empty(0)

        # splitting the data into 2 subsets
        first_instance = True 
        for idx, el in enumerate(y):
            if self.larger_than_boundary:
                if first_instance:
                    if (el > self.boundary):
                        y1 = np.append(y1, el)
                        x1 = np.append(x1, x[idx])
                    else:
                       y2 = np.append(y2, el)
                       x2 = np.append(x2, x[idx])
                       first_instance = False
                else:
                    y2 = np.append(y2, el)
                    x2 = np.append(x2, x[idx])
            else:
                if first_instance:
                    if (el < self.boundary):
                        y1 = np.append(y1, el)
                        x1 = np.append(x1, x[idx])
                    else:
                        y2 = np.append(y2, el)
                        x2 = np.append(x2, x[idx])
                        first_instance = False  
                else:
                    x2 = np.append(x2, x[idx])
                    y2 = np.append(y2, el)
                  
        # fitting functions
        if self.linear_first:
            # take 1 point from the linear part of the graph to put in the curve fitting
            y2 = np.append(y1[-1], y2)
            x2 = np.append(x1[-1], x2)
            y1 = y1[:-1]
            x1 = x1[:-1]
            if fitting_function.lower() == "cubic":
                print(f"y2: {y2}")
                fth = FitCubicPoly(x2,y2).fit()
            elif fitting_function.lower() == "biquad":
                try:
                    fth = FitBiquadPoly(x2,y2).fit()
                except Exception as e:
                    print(e)
                    return
            else:
                fth = FitTanh(x2,y2).fit()
                print(f"y2: {y2}")
            reg = LinearRegression().fit(x1.reshape(-1,1), y1.reshape(-1,1))
        else:
            # take 1 point from the linear part of the graph to put in the curve fitting
            y1 = np.append(y1, y2[1])
            x1 = np.append(x1, x2[1])
            y1 = y1[1:]
            x1 = x1[1:]
            if fitting_function.lower() == "cubic":
                fth = FitCubicPoly(x1,y1).fit()
            elif fitting_function.lower() == "biquad":
                try:
                    fth = FitBiquadPoly(x1,y1).fit()
                except Exception as e:
                    print(e)
                    return
            else:
                fth = FitTanh(x1,y1).fit()
            reg = LinearRegression().fit(x2.reshape(-1,1), y2.reshape(-1,1))
            print(f"y2_linear: {y2}")
        
        # # predictions
        yq_lin = reg.predict(xq.reshape(-1,1))
        yq_tanh = fth.predict(xq)
        yq = np.abs(yq_lin.flatten() - yq_tanh.flatten())
        idx = np.argmin(yq)
        zero_pt = xq[idx]
        print('zero point {}'.format(round(zero_pt,3)))
        self.xq = xq
        self.yq = yq
        self.val = yq[idx]
        self.vol = zero_pt
        self.yq_lin = yq_lin
        self.yq_tanh = yq_tanh
        # fitting parameters
        self.fth = fth
        self.reg = reg
        self.success = True
        
        
    def plots(self, fitting_function):
        if self.success:
            x, y, y_error, xq, yq, yq_lin, yq_tanh = self.x, self.y, self.y_error, self.xq, self.yq, self.yq_lin, self.yq_tanh 
            val, vol = self.val, self.vol
            fontsize = 18
            legendsize = 15
            plt.figure(figsize=(6,4))
            plt.scatter(x,y, s=14, c = 'r')#marker= '.')#, 'r*')
            plt.errorbar(x,y, yerr = y_error, fmt = 'o', ecolor = 'blue', elinewidth = 2, capsize = 5, mfc = 'red', mec ='red')
            plt.plot(xq,yq_lin, 'g--', )
            plt.plot(xq,yq_tanh, 'b--')
            plt.xlabel('volume [$\mu$l]',fontsize=fontsize)
            plt.ylabel(self.params,fontsize=fontsize)
            plt.title('vol: '+str(round(vol,3))+' $\mu$l|val: '+str(round(val)),fontsize=fontsize)
            ax = plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.tick_params(axis='both', which='minor', labelsize=fontsize)
            ax.set_aspect('auto')
            # ax.legend(['raw data','lin fit','tanh fit'], fontsize = fontsize)
            plt.ylim(min(y)-30, max(y)+20)
            ax.legend(['raw data','lin fit',f'{fitting_function} fit'], fontsize = legendsize)
            plt.savefig(os.path.join(self.save_path,f'{fitting_function} fit.jpg'),bbox_inches = 'tight')
            plt.show()
            plt.close()
        else:
            return None

        
    def __save_pickle(self):
        with open(os.path.join(self.save_path, 'fit_data.pkl'), 'wb') as f:
            pickle.dump(self, f)
       
           
    def __save_csv(self):
        df = pd.DataFrame({'vol': [self.vol],
                           'val':[self.val],
                           'para_a': [self.fth.para.x[2]],
                           'para_x0': [self.fth.para.x[0]],
                           'para_b': [self.fth.para.x[1]],
                           'intercept': self.reg.coef_[0],
                           'coef': self.reg.intercept_})            
        df.to_csv(os.path.join(self.save_path,'fit_result.csv'))
        