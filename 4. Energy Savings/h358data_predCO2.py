import matplotlib
import pandas
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
import timemg
from math import log10
import solar
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn import linear_model
import numpy as np


__author__ = 'stephane.ploix@g-scop.grenoble-inp.fr'


class DataContainer:

    def __init__(self, csv_filename: str):
        self.sample_time = None
        self.starting_stringdatetime = None
        self.ending_stringdatetime = None
        self.registered_databases = dict()
        self.data = dict()
        self.extracted_variables = list()
        self.data['stringtime'] = None
        self.extracted_variables.append('stringtime')
        self.data['epochtime'] = None
        self.extracted_variables.append('epochtime')
        self.data['datetime'] = None
        self.extracted_variables.append('datetime')

        self.SPECIAL_VARIABLES = ('stringtime', 'epochtime', 'datetime')
        self.variable_full_name_id_dict = dict()  # context$variable: variable_id
        self.variable_full_name_database_dict = dict()  # context$variable: database_name
        self.variable_full_name_csv_dict = dict()  # context$variable: csv_file_name
        self.variable_type = dict()
        self.contexts = list()

        self.data = dict()
        self._extracted_variable_full_names = list()  # context_variable names
        self.data['stringtime'] = None
        self._extracted_variable_full_names.append('stringtime')
        self.data['epochtime'] = None
        self._extracted_variable_full_names.append('epochtime')
        self.data['datetime'] = None
        self._extracted_variable_full_names.append('datetime')

        dataframe = pandas.read_csv(csv_filename, dialect='excel')
        variable_names = dataframe.columns
        for variable_name in variable_names:
            if variable_name == 'stringtime':
                dataframe['stringtime'].astype({'stringtime': 'str'})
                self.data[variable_name] = dataframe[variable_name].values.tolist()
            elif variable_name == 'datetime':
                self.data[variable_name] = [
                    timemg.stringdate_to_datetime(stringdatetime, date_format='%Y-%m-%d %H:%M:%S') for stringdatetime in dataframe['datetime']]
            elif variable_name == 'epochtime':
                dataframe[variable_name].astype({'epochtime': 'int'})
                self.data[variable_name] = dataframe[variable_name].values.tolist()
            else:
                self.add_external_variable(variable_name, dataframe[variable_name].values.tolist())


        self.starting_stringdatetime = self.data['stringtime'][0]
        self.ending_stringdatetime = self.data['stringtime'][-1]
        self.sample_time_in_secs = int((self.data['epochtime'][1]-self.data['epochtime'][0]) / 1000)

    def add_external_variable(self, label: str, values: list):
        if label not in self.extracted_variables:
            self.data[label] = values
            self.extracted_variables.append(label)
        else:
            print('variable %s already extracted' % label)

    def get_variable(self, label: str):
        return self.data[label]

    def get_number_of_variables(self):
        return len(self.extracted_variables)

    def get_number_of_samples(self):
        if self.data['epochtime'] is None:
            return 0
        else:
            return len(self.data['epochtime'])

    def _plot_selection(self, int_vars: list):
        styles = ('-', '--', '-.', ':')
        linewidths = (3.0, 2.5, 2.5, 1.5, 1.0, 0.5, 0.25)
        figure, axes = plt.subplots()
        axes.set_title('from %s to %s' % (self.starting_stringdatetime, self.ending_stringdatetime))
        text_legends = list()
        for i in range(len(int_vars)):
            if int_vars[i].get():
                style = styles[i % len(styles)]
                linewidth = linewidths[i // len(styles) % len(linewidths)]
                time_data = list(self.data['datetime'])
                value_data = list(self.data[self.extracted_variables[i + 3]])
                if len(time_data) > 1:
                    time_data.append(time_data[-1] + (time_data[-1] - time_data[-2]))
                    value_data.append(value_data[-1])
                axes.step(time_data, value_data, linewidth=linewidth, linestyle=style, where='post')
                axes.set_xlim([time_data[0], time_data[-1]])
                text_legends.append(self.extracted_variables[i + 3])
                int_vars[i].set(0)
        axes.legend(text_legends, loc=0)
        figure.set_tight_layout(True)
        axes.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        axes.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        axes.xaxis.set_minor_locator(mdates.DayLocator())
        axes.grid(True)
        plt.show()

    def plot(self):
        tk_variables = list()
        tk_window = tk.Tk()
        tk_window.wm_title('variable plotter')
        tk.Button(tk_window, text='plot', command=lambda: self._plot_selection(tk_variables)).grid(row=0, column=0, sticky=tk.W + tk.E)
        frame = tk.Frame(tk_window).grid(row=1, column=0, sticky=tk.N + tk.S)
        vertical_scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
        vertical_scrollbar.grid(row=1, column=1, sticky=tk.N + tk.S)
        canvas = tk.Canvas(frame, width=400, yscrollcommand=vertical_scrollbar.set)
        tk_window.grid_rowconfigure(1, weight=1)
        canvas.grid(row=1, column=0, sticky='news')
        vertical_scrollbar.config(command=canvas.yview)
        checkboxes_frame = tk.Frame(canvas)
        checkboxes_frame.rowconfigure(1, weight=1)
        for i in range(3, len(self.extracted_variables)):
            tk_variable = tk.IntVar(0)
            tk_variables.append(tk_variable)
            tk.Checkbutton(checkboxes_frame, text=self.extracted_variables[i], variable=tk_variable, offvalue=0).grid(row=(i - 3), sticky=tk.W)
        canvas.create_window(0, 0, window=checkboxes_frame)
        checkboxes_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))
        tk_window.geometry(str(tk_window.winfo_width()) + "x" + str(tk_window.winfo_screenheight()))
        tk_window.mainloop()

    def __str__(self):
        string = 'Data cover period from %s to %s with time period: %d seconds\nRegistered database:\n' % (self.starting_stringdatetime, self.ending_stringdatetime, self.sample_time)
        for database in self.registered_databases:
            string += '- %s \n' % database
        string += 'Available variables:\n'
        for variable_name in self.extracted_variables:
            string += '- %s \n' % variable_name
        return string


if __name__ == '__main__':
    h358 = DataContainer('h358data_winter2015-2016.csv')
    CCO2 = h358.get_variable('office_CO2_concentration')
    Toffice_reference = h358.get_variable('Toffice_reference')
    Tout = h358.get_variable('Tout')
    Tcorridor = h358.get_variable('Tcorridor')
    occupancy = h358.get_variable('occupancy')
    window_opening = h358.get_variable('window_opening')
    door_opening = h358.get_variable('door_opening')
    dT_heat = h358.get_variable('dT_heat')
    CCO2_n = h358.get_variable('corridor_CO2_concentration')
    humidity_outdoor = h358.get_variable('humidity_outdoor')
    nebulosity = h358.get_variable('nebulosity')
    datetime = h358.get_variable('datetime')
    power_stephane = h358.get_variable('power_stephane')
    power_khadija = h358.get_variable('power_khadija')
    power_audrey = h358.get_variable('power_audrey')
    power_stagiaire = h358.get_variable('power_stagiaire')
    power_block_east = h358.get_variable('power_block_east')
    power_block_west = h358.get_variable('power_block_west')
    power_heater =h358.get_variable('power_heater')
    actual_occupation = h358.get_variable('actual_occupation')


    ### put your code about waste calculation here  (useful variables: Toffice_reference, Tout, Tcorridor, dT_heat, CCO2, zetaW, zetaD)
    window_waste = []
    door_waste = []
    for k in range (0,len(Tout)):
        if CCO2[k]<1000 and dT_heat[k]>0 and occupancy[k]>0 and Tout[k]<Toffice_reference[k]:
            window_waste.append((Toffice_reference[k]-Tout[k])*(window_opening[k]))
        else:
            window_waste.append(0)
    print(window_waste)
    for k in range (0,len(Tout)):
        if CCO2[k] < 1000 and dT_heat[k] > 0 and occupancy[k] > 0 and Tcorridor[k] < Toffice_reference[k]:
            door_waste.append((Toffice_reference[k] - Tcorridor[k]) * (door_opening[k]) )
        else:
            door_waste.append(0)

    #door_waste = (Toffice_reference-Tcorridor)*(door_opening)*dT_heat*CCO2
    h358.add_external_variable('window_waste', window_waste)  # uncomment when ready
    h358.add_external_variable('door_waste', door_waste)  # uncomment when ready

    # ICONE containment indicator (use occupancy and CCO2)
    CK = []
    f1 = 0
    f2= 0
    i = 0
    for k in range(0, len(Tout)):
        if occupancy[k]==0:
            CK.append(-1)
        if occupancy[k] > 0 and (CCO2[k]<1000):
            CK.append(0)
            i +=1
        if occupancy[k] >0 and ((CCO2[k]<1700) and(CCO2[k] >= 1000)) :
            CK.append(1)
            f1 +=1
            i+=1
        if occupancy[k] > 0 and (CCO2[k] >= 1700):
            CK.append(2)
            f2 +=1
            i+=1
    f1 = f1/i
    f2 = f2/i

    ICONE = 8.3*log10(1+f1+3*f2)
    print(ICONE)


    Cin = []
    V = 7*4*2.5
    Qw0 = V/(2*3600)
    for k in range(0, len(Tout)):
        Qw = Qw0*(1+2*window_opening[k])
        Qd = Qw0 * (1 + 2*door_opening[k])
        Cin.append(((Qw*400)+Qd*CCO2_n[k]+6.5*occupancy[k])/(Qw+Qd))
    h358.add_external_variable('CO2_calc', Cin)

    solar_gain = solar.SolarGain()

    sol = []
    for k in range(0, len(Tout)):
        sol.append(solar_gain.get_solar_gain(solar.SOUTH, solar.VERTICAL, datetime[k], temperature=Tout[k],
                                    humidity=humidity_outdoor[k] / 100,
                                    nebulosity_in_percentage=nebulosity[k] / 100,
                                    pollution=0.1)[0])
    h358.add_external_variable('sol', sol)

    occ = []

    for k in range(0, len(Tout)):

        a,b,c,d = 0,0,0,0
        if power_stephane[k] >= 17:
            a =1
        if power_khadija[k] >= 17:
            b =1
        if power_audrey[k] >= 17:
            c=1
        if power_stagiaire[k] >= 17:
            d=1
        occ.append(a+b+c+d)

    h358.add_external_variable('occ', occ)

    tot_elec = []

    for k in range(0, len(Tout)):
        tot_elec.append(power_block_east[k] + power_block_west[k])

    h358.add_external_variable('tot_elec', tot_elec)


    internal_gain = []

    for k in range(0, len(Tout)):
        internal_gain.append(occupancy[k]*100+dT_heat[k]*30)
    h358.add_external_variable('internal_gain', internal_gain)

    Tsim =[]
    corridor_resistance = 0.0338
    out_resistance = 0.0228
    down_resistance = 0.0376


    for k in range(0, len(Tout)):

        RW = 1 / (1.2 * 1 * (Qw0 + 2 * Qw0 * window_opening[k]))
        RD = 1 / (1.2 * 1 * (Qw0 + 10* Qw0 * door_opening[k]))


        out_resistanc = 1/((1/out_resistance)+(1/RW))
        corridor_resistanc = 1 / ((1 / corridor_resistance) + (1 / RD))

        Tsim.append((((internal_gain[k]+tot_elec[k]+0.3*sol[k])+(Tcorridor[k]/corridor_resistanc)+(Tout[k]/out_resistanc))/((1/out_resistanc)+(1/corridor_resistanc))))

    h358.add_external_variable('Tsim', Tsim)

    #h358.plot()


    Tsens0 = []

    for out_resistance in [0.01,0.015,0.020,0.0228,0.025]:
        Tsens = []
        for k in range(0, len(Tout)):
            RW = 1 / (1.2 * 1 * (Qw0 + 2 * Qw0 * window_opening[k]))
            RD = 1 / (1.2 * 1 * (Qw0 + 10 * Qw0 * door_opening[k]))

            out_resistanc = 1 / ((1 / out_resistance) + (1 / RW))
            corridor_resistanc = 1 / ((1 / corridor_resistance) + (1 / RD))

            Tsens.append((((internal_gain[k] + tot_elec[k] + 0.3 * sol[k]) + (Tcorridor[k] / corridor_resistanc) + (
                        Tout[k] / out_resistanc)) / ((1 / out_resistanc) + (1 / corridor_resistanc))))
        Tsens0.append(Tsens)






    # plt.plot(Tsens0[0])
    # plt.plot(Tsens0[1])
    # plt.plot(Tsens0[2])
    # plt.plot(Tsens0[3])
    # plt.plot(Tsens0[4])
    # plt.show()
    #y = np.array(Toffice_reference)
    #X = np.array(occupancy)
    df = pandas.DataFrame(np.array(Toffice_reference),columns=["label"])
    df['CCO2']= np.array(CCO2)
    df['occupancy']=np.array(occupancy)
    #df['occ'] = np.array(occ)
    df['Tout']=np.array(Tout)
    df['power_heater'] = np.array(power_heater)

    df['nebulosity']=np.array(nebulosity)
    #df['internal_gain']=np.array(internal_gain)
    df['window_opening'] = np.array(window_opening)
    df['door_opening']=np.array(door_opening)
    #df['humidity_outdoor'] = np.array(humidity_outdoor)
    #df['door_waste'] = np.array(door_waste)
    #df['window_waste'] = np.array(window_waste)
    #df['dt_heat'] = np.array(dT_heat)
    #df['humidity_outdoor'] = np.array(humidity_outdoor)
    df['Tempp_1'] = df['label'].shift(1)
    df['Tempp_1'][0] = df['label'][0]

    df['CCO2_1'] = df['CCO2'].shift(1)
    df['CCO2_1'][0] = df['CCO2'][0]

    df['Q_1'] = df['power_heater'].shift(1)
    df['Q_1'][0] = df['power_heater'][0]

    df['window_opening_1'] = df['window_opening'].shift(1)
    df['window_opening_1'][0] = df['window_opening'][0]

    #df.plot()
    #plt.show()
    print(df.columns)
    # #X= np.array([CCO2,occupancy,Tout,window_opening,door_opening,nebulosity,humidity_outdoor,sol,internal_gain,tot_elec])
    #
    x_t = df.drop(['label','CCO2','CCO2_1'],axis=1)
    model_T = linear_model.Lasso(alpha=1.0)
    # # fit model
    print(np.array([occ, power_heater]).shape)
    print(np.array(Toffice_reference).reshape)
    model_T.fit(df.drop(['label','CCO2','CCO2_1'],axis=1), np.array(Toffice_reference))
    # # define new data
    #
    # # make a prediction
    yhat = model_T.predict(x_t)
    # # summarize prediction
    # #print('Predicted: %.3f' % yhat)
    #plt.plot(yhat,color="pink")
    #plt.plot(df['label'])
    #plt.show()
    print(yhat)

    x_q = df.drop(['power_heater','CCO2','CCO2_1'], axis=1)
    model_Q = linear_model.Lasso(alpha=1.0)
    # # fit model
    model_Q.fit(df.drop(['power_heater','CCO2','CCO2_1'],axis=1), np.array(power_heater))
    # # define new data
    #
    # # make a prediction

    yhat_Q = model_Q.predict(x_q)
    # # summarize prediction
    # #print('Predicted: %.3f' % yhat)
    plt.plot(yhat_Q,color="pink")
    plt.show()
    plt.plot(df['CCO2'].divide(100))
    plt.plot(df['window_opening'])
    plt.plot(df['label'])
    plt.show()
    print(yhat_Q)

    print(max(yhat_Q))


    x_CCO2 = np.array([[df['occupancy']],[df['window_opening']],[df['door_opening']],[df['CCO2_1']]]).transpose()
    model_CCO2 = linear_model.Lasso(alpha=1.0)
    # # fit model
    model_CCO2.fit(df[['occupancy','window_opening','door_opening','CCO2_1']], np.array(CCO2))
    # # define new data
    #
    # # make a prediction

    yhat_CCO2 = model_CCO2.predict(df[['occupancy','window_opening','door_opening','CCO2_1']])
    # # summarize prediction
    # #print('Predicted: %.3f' % yhat)
    #plt.plot(yhat_CCO2,color="pink")
    #plt.plot(df['CCO2'])
    #plt.show()
    print(yhat_CCO2)


    #
    # def heater_energy():
    #     energy = 0
    #     for k in range(power_heater):
    #         energy += k
    #     return k
    #
    Tcomfort= 22
    Tcomfort_min = 20
    Tsetback = 15
    #print(model_T.predict(np.array([[0], [900.0]]).transpose()))

    df['window_opening2'] = df['window_opening'].clip(upper=0)
    df['occupancy'].plot()
    plt.show()
    def control(t_window):
        k =0
        T = []
        T.append(Tsetback)
        q=[]
        q.append(0)

        df['window_opening2'][t_window:] = df['window_opening'][t_window:].clip(lower=0.5)
        plt.plot(df['window_opening2'])
        plt.show()

        while k<2000:
            if df['occupancy'][k+1] >= 0.5 :
                Tset = Tcomfort
                Q = model_Q.predict(np.array(
                    [[Tset], [df['occupancy'][k + 1]], [df['Tout'][k + 1]], [df['nebulosity'][k + 1]],[df['window_opening2'][k+1]],[df['door_opening'][k + 1]],
                     [T[k]],[q[k]],[df['window_opening2'][k]]]).transpose())
                if Q > 800:
                    print("kk")
                    Tdiscomfort = model_T.predict(np.array(
                        [[df['occupancy'][k + 1]], [df['Tout'][k + 1]], [800.0], [df['nebulosity'][k + 1]],
                         [df['window_opening2'][k]], [df['door_opening'][k + 1]], [T[k]],[q[k]],[df['window_opening2'][k]]]).transpose())
                   # q.append(model_Q.predict(np.array(
                    #    [[Tdiscomfort], [df['occupancy'][k + 1]], [df['Tout'][k + 1]], [df['nebulosity'][k + 1]],[df['window_opening2'][k + 1]],[df['door_opening'][k + 1]],
                     #    [T[k]]]).transpose()))
                    q.append(800)
                    T.append(Tdiscomfort)
                else:
                    q.append(Q)
                    T.append(Tset)




            else:
                n_horizon = 1
                while df['occupancy'][k+n_horizon] <= 0.5:
                    n_horizon+=1
                print(n_horizon)
                n_preheat =0 # "time form Tair_noheat to setback"
                Tnow = T[k]

                while Tnow < Tcomfort_min:
                    n_preheat +=1
                    print(n_preheat)
                    Tnow = model_T.predict(np.array([[df['occupancy'][k+n_preheat]],[df['Tout'][k+n_preheat]], [800],[df['nebulosity'][k+n_preheat]],[df['window_opening2'][k + n_preheat]],[df['door_opening'][k + n_preheat]],
                                                     [Tnow],[800],[df['window_opening2'][k+n_preheat-1]]]).transpose())
                    print(Tnow)
                if n_horizon >= n_preheat:
                    if model_T.predict(np.array([[df['occupancy'][k+1]],[df['Tout'][k+1]], [0],[df['nebulosity'][k+1]],[df['window_opening2'][k + 1]],[df['door_opening'][k + 1]],[T[k]],[q[k]],[df['window_opening2'][k]]]).transpose())> Tsetback:
                        T.append(model_T.predict(np.array([[df['occupancy'][k+1]],[df['Tout'][k+1]], [0],[df['nebulosity'][k+1]],[df['window_opening2'][k + 1]],[df['door_opening'][k + 1]],[T[k]],[q[k]],[df['window_opening2'][k]]]).transpose()))
                        q.append(0)
                    else:
                        Tset = Tsetback
                        T.append(Tset)
                        q.append(model_Q.predict(np.array([[Tset],[df['occupancy'][k+1]],[df['Tout'][k+1]],[df['nebulosity'][k+1]],[df['window_opening2'][k + 1]],[df['door_opening'][k + 1]],[T[k]],[q[k]],[df['window_opening2'][k]]]).transpose()))
                else:
                    Tset = Tcomfort
                    q.append(max(yhat_Q))
                    T.append(model_T.predict(np.array([[df['occupancy'][k+1]],[df['Tout'][k+1]],[800],[df['nebulosity'][k+1]],[df['window_opening2'][k + 1]],[df['door_opening'][k + 1]],[T[k]],[q[k]],[df['window_opening2'][k]]]).transpose()))
            k = k+1
            print(k)

        return T,q
    result = control(245)
    plt.plot(result[0])
    plt.plot(result[1])
    #plt.plot(Toffice_reference[0:2000])
    plt.plot(occupancy[0:2000])
    #plt.plot(power_heater[0:2000])
    #plt.plot(CCO2[0:2000])
    plt.show()

    print(sum(result[1]))
    print(sum(power_heater[0:2000]))
    #
    #
    #
    #
    #
    #
    #
    # def optimize(self):
    #     V = lambda x: 0.016 * x[2] + 4 * x[0] * x[1] * x[2]
    #     bnds = ((0.03, 0.05), (0.1, 0.2), (0.1, 0.30))
    #     x0 = np.array([self.w_leg, self.h_leg, self.depth])
    #     res = opt.minimize(V, x0, method='SLSQP', jac= '2-point',bounds=bnds, constraints={"fun": self.constraint, "type": "ineq"},options={'maxiter': 100})
    #     print(res.x)
    #     self.w_leg = res.x[0]
    #     self.h_leg = res.x[1]
    #     self.depth = res.x[2]
    #     print(V(res.x))
    #     #self.build()
    #
    # def constraint(self,x):
    #     self.w_leg = x[0]
    #     self.h_leg = x[1]
    #     self.depth = x[2]
    #     self.build()
    #     #self.mesh.refine()
    #     self.solve()
    #     T = max(self.solve())
    #     print(T)
    #     return (self.Tmax-T)
    #
    #
    #
    # def solve(self):
    #         # make problem and solve it
    #         solution = linalg.solve(Make(self.mesh).matri()[0], Make(self.mesh).matri()[1])
    #         i = 0
    #         for k in self.mesh.nodes:
    #             if k.dirichlet == False:
    #                 k.value = solution[i]
    #                 i += 1
    #         i = 0
    #
    #
    #         print(max(solution))
    #         return (solution)
    #
    #
    #
    #
    #
    #
    #
