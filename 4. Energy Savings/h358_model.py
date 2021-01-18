import math
import numpy
import scipy.constants
import solar
from h358data import DataContainer


corridor_resistance= 0.0338
out_resistance=  0.0228
down_resistance= 0.0376

# variables
h358 = DataContainer('h358data_2015-2016.csv')
datetime = h358.get_variable('datetime')
office_CO2_concentration = h358.get_variable('office_CO2_concentration')
corridorCCO2 = h358.get_variable('corridor_CO2_concentration')
Toffice_reference = h358.get_variable('Toffice_reference')
Tcorridor = h358.get_variable('Tcorridor')
humidity_outdoor = h358.get_variable('humidity_outdoor')
nebulosity = h358.get_variable('nebulosity')
Tout = h358.get_variable('Tout')
power_stephane = h358.get_variable('power_stephane')
power_khadija = h358.get_variable('power_khadija')
power_audrey = h358.get_variable('power_audrey')
power_stagiaire = h358.get_variable('power_stagiaire')
power_block_east = h358.get_variable('power_block_east')
power_block_west = h358.get_variable('power_block_west')
window_opening = h358.get_variable('window_opening')
door_opening = h358.get_variable('door_opening')
dT_heat = h358.get_variable('dT_heat')

solar_gain = solar.SolarGain()
phi_sun = []
## TO BE COMPLETED

print(solar_gain.get_solar_gain(solar.SOUTH, solar.VERTICAL, datetime[k], temperature=Tout[k], humidity=humidity_outdoor[k] / 100,
                                nebulosity_in_percentage=nebulosity[k] / 100,
                                pollution=0.1))

Cin = []
V = 7 * 4 * 2.5
Qw0 = V / (2 * 3600)
for k in range(0, len(Tout)):
    Qw = Qw0 * (1 + 2 * window_opening[k])
    Qd = Qw0 * (1 + 2 * door_opening[k])
    Cin.append(((Qw * 400) + Qd * CCO2_n[k] + 6.5 * occupancy[k]) / (Qw + Qd))
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

    a, b, c, d = 0, 0, 0, 0
    if power_stephane[k] >= 17:
        a = 1
    if power_khadija[k] >= 17:
        b = 1
    if power_audrey[k] >= 17:
        c = 1
    if power_stagiaire[k] >= 17:
        d = 1
    occ.append(a + b + c + d)

h358.add_external_variable('occ', occ)

tot_elec = []

for k in range(0, len(Tout)):
    tot_elec.append(power_block_east[k] + power_block_west[k])

h358.add_external_variable('tot_elec', tot_elec)

internal_gain = []

for k in range(0, len(Tout)):
    internal_gain.append(occupancy[k] * 100 + dT_heat[k] * 30)
h358.add_external_variable('internal_gain', internal_gain)

Tsim = []
corridor_resistance = 0.0338
out_resistance = 0.0228
down_resistance = 0.0376
## To BE COMPLETEDD

#static thermal model


def simulate(door_opening_forced: bool=None, window_opening_forced: bool=None, Tout_bias: float=0, Tcorridor_bias: float=0, office_power_gains_bias: float=0):
    office_simulated_temperature = []
    office_simulated_CO2 = []
    for k in range(0, len(Tout)):

        RW = 1 / (1.2 * 1 * (Qw0 + 2 * Qw0 * window_opening[k]))
        RD = 1 / (1.2 * 1 * (Qw0 + 10* Qw0 * door_opening[k]))


        out_resistanc = 1/((1/out_resistance)+(1/RW))
        corridor_resistanc = 1 / ((1 / corridor_resistance) + (1 / RD))

        office_simulated_temperature.append((((internal_gain[k]+tot_elec[k]+0.3*sol[k])+(Tcorridor[k]/corridor_resistanc)+(Tout[k]/out_resistanc))/((1/out_resistanc)+(1/corridor_resistanc))))

    h358.add_external_variable('Tsim', Tsim)

    return office_simulated_temperature, office_simulated_CO2


office_simulated_temperature_ref, office_simulated_CO2_ref = simulate()
h358.add_external_variable('office_simulated_temperature', office_simulated_temperature_ref)
h358.add_external_variable('office_simulated_CO2', office_simulated_CO2_ref)

office_simulated_temperature, office_simulated_CO2 = simulate(door_opening_forced=True)
h358.add_external_variable('office_simulated_temperature_door_opening_forced', office_simulated_temperature)
h358.add_external_variable('office_simulated_CO2_door_opening_forced', office_simulated_CO2)
print('_____ door opening forced ____')
print('temperature error:', sum([abs(office_simulated_temperature_ref[k]-office_simulated_temperature[k]) for k in range(len(datetime))])/len(datetime))
print('CO2 error:', sum([abs(office_simulated_CO2_ref[k]-office_simulated_CO2[k]) for k in range(len(datetime))])/len(datetime))

office_simulated_temperature, office_simulated_CO2 = simulate(window_opening_forced=True)
h358.add_external_variable('office_simulated_temperature_window_opening_forced', office_simulated_temperature)
h358.add_external_variable('office_simulated_CO2_window_opening_forced', office_simulated_CO2)
print('_____  window opening forced ____')
print('Temperature error error:', sum([abs(office_simulated_temperature_ref[k]-office_simulated_temperature[k]) for k in range(len(datetime))])/len(datetime))
print('CO2 error:', sum([abs(office_simulated_CO2_ref[k]-office_simulated_CO2[k]) for k in range(len(datetime))])/len(datetime))

office_simulated_temperature, office_simulated_CO2 = simulate(Tout_bias=5)
h358.add_external_variable('office_simulated_temperature_Tout_bias5', office_simulated_temperature)
h358.add_external_variable('office_simulated_CO2_Tout_bias5', office_simulated_CO2)
print('_____  Tout bias ____')
print('Temperature error:', sum([abs(office_simulated_temperature_ref[k]-office_simulated_temperature[k]) for k in range(len(datetime))])/len(datetime))
print('CO2 error:', sum([abs(office_simulated_CO2_ref[k]-office_simulated_CO2[k]) for k in range(len(datetime))])/len(datetime))

office_simulated_temperature, office_simulated_CO2 = simulate(Tcorridor_bias=5)
h358.add_external_variable('office_simulated_temperature_Tcorridor_bias5', office_simulated_temperature)
h358.add_external_variable('office_simulated_CO2_Tcorridor_bias5', office_simulated_CO2)
print('_____  Tcorridor bias ____')
print('Temperature error:', sum([abs(office_simulated_temperature_ref[k]-office_simulated_temperature[k]) for k in range(len(datetime))])/len(datetime))
print('CO2 error:', sum([abs(office_simulated_CO2_ref[k]-office_simulated_CO2[k]) for k in range(len(datetime))])/len(datetime))

office_simulated_temperature, office_simulated_CO2 = simulate(office_power_gains_bias=100)
h358.add_external_variable('office_simulated_temperature_office_power_gains_bias100', office_simulated_temperature)
h358.add_external_variable('office_simulated_CO2_office_power_gains_bias100', office_simulated_CO2)
print('_____  gains bias ____')
print('Temperature error power:', sum([abs(office_simulated_temperature_ref[k]-office_simulated_temperature[k]) for k in range(len(datetime))])/len(datetime))
print('CO2 error power:', sum([abs(office_simulated_CO2_ref[k]-office_simulated_CO2[k]) for k in range(len(datetime))])/len(datetime))
h358.plot()


