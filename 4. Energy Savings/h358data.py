import matplotlib
import pandas
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tkinter as tk
import timemg
from math import log10
import solar

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
    power_heater = h358.get_variable('power_heater')

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

    h358.plot()


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




    plt.plot(Tsens0[0])
    plt.plot(Tsens0[1])
    plt.plot(Tsens0[2])
    plt.plot(Tsens0[3])
    plt.plot(Tsens0[4])
    plt.show()


