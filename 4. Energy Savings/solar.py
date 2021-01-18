__author__ = 'stephane.ploix@g-scop.grenoble-inp.fr'

from math import pi, cos, sin, acos, asin, atan, sqrt, exp, log
from typing import TypeVar

import matplotlib.pyplot as plt
from pytz import timezone, utc

import timemg

Datetime = TypeVar('datetime')

SOUTH = pi
EAST = - pi / 2
WEST = pi / 2
NORTH = 0

HORIZONTAL = 0  # solar captor directed horizontally to the sky zenith
VERTICAL = pi / 2  # solar captor directed vertically


def sign(x: float):
    return 1 if x > 0 else -1 if x < 0 else 0


class SolarGain():

    def __init__(self, time_zone: str='Europe/Paris', latitude_in_deg: float=45.183, longitude_in_deg: float=5.717, sea_level_in_meters: float=330, albedo: float=0.1):
        """
        compute solar gains at a given location (Grenoble by default)
        :param time_zone: time zone
        :param latitude_in_deg: latitude in degrees
        :param longitude_in_deg: longitude in degrees
        :param sea_level_in_meters: sea level in meters
        :param albedo: albedo
        """
        self.time_zone = time_zone
        self.latitude_in_deg = latitude_in_deg
        self.longitude_in_deg = longitude_in_deg
        self.sea_level_in_meters = sea_level_in_meters
        self.albedo = albedo

    def get_solar_time(self, datetime: Datetime):
        """
        convert administrative time to solar time
        :param datetime: datetime to be converted
        :return: day_in_year, solartime_in_secondes
        """
        thetimezone = timezone(self.time_zone)
        local_datetime = thetimezone.localize(datetime, is_dst=True)
        utc_datetime = local_datetime.astimezone(utc)
        utc_timetuple = utc_datetime.timetuple()
        day_in_year = utc_timetuple.tm_yday
        hour_in_day = utc_timetuple.tm_hour
        minute_in_hour = utc_timetuple.tm_min
        seconde_in_minute = utc_timetuple.tm_sec
        standard_time_in_seconds = hour_in_day * 3600 + minute_in_hour * 60 + seconde_in_minute + self.longitude_in_deg * 4 * 60
        coefsin1 = 0.001868
        coefcos1 = 0.032077
        angle1 = atan(coefsin1 / coefcos1)
        deltaday1 = angle1 / (2 * pi) * 365
        coef1 = sqrt(coefcos1 ** 2 + coefsin1 ** 2)
        coefsin2 = 0.014615
        coefcos2 = 0.04089
        angle2 = atan(coefsin2 / coefcos2)
        coef2 = sqrt(coefcos2 ** 2 + coefsin2 ** 2)
        deltaday2 = angle2 / (4 * pi) * 365
        solartime_in_secondes = standard_time_in_seconds + 229.2 * 60 * (0.000075 + coef1 * sin(2 * pi * (deltaday1 - day_in_year) / 365) - coef2 * sin(4 * pi * (deltaday2 + day_in_year) / 365))
        return day_in_year, solartime_in_secondes

    def get_solar_angles(self, day_in_year: int, solartime_in_secondes: int):
        """
        calculate angles
        :param day_in_year: day of year between 0 and 265
        :param solartime_in_secondes: solar time in seconds
        :return: altitude_in_rad, azimuth_in_rad, hour_angle_in_rad, latitude_in_rad, declination_in_rad
        """
        latitude_in_rad = self.latitude_in_deg / 180 * pi
        declination_in_rad = 23.45 * pi / 180 * sin(2 * pi * (285 + day_in_year) / 365)
        hour_angle_in_rad = pi / 12 * (solartime_in_secondes / 3600 - 12)
        altitude_in_rad = asin(sin(declination_in_rad) * sin(latitude_in_rad) + cos(declination_in_rad) * cos(latitude_in_rad) * cos(hour_angle_in_rad))
        if 0 < altitude_in_rad < pi/2 :
            azimuth_in_rad = sign(hour_angle_in_rad) * abs(acos((sin(altitude_in_rad) * sin(latitude_in_rad) - sin(declination_in_rad)) / (cos(altitude_in_rad) * cos(latitude_in_rad))))
        else:
            altitude_in_rad = 0
            azimuth_in_rad = 0
        return altitude_in_rad, azimuth_in_rad, hour_angle_in_rad, latitude_in_rad, declination_in_rad

    def get_solar_beam_transfer(self, phis_with_nebulosity: float, altitude_in_rad: float, temperature: float, humidity: float, pollution: float):
        atmospheric_pressure = 101325 * (1 - 2.26e-5 * self.sea_level_in_meters) ** 5.26
        transmitivity = 0.6 ** ((sqrt(1229 + (614 * sin(altitude_in_rad)) ** 2) - 614 * sin(altitude_in_rad)) * ((288 - 0.0065 * self.sea_level_in_meters) / 288) ** 5.256)
        air_mass = atmospheric_pressure / (101325 * sin(altitude_in_rad) + 15198.75 * (3.885 + altitude_in_rad) ** (-1.253))
        Erayleigh = 1 / (0.9 * air_mass + 9.4)
        Pv = 2.165 * (1.098 + temperature / 100) ** 8.02 * humidity
        lLinke = 2.4 + 14.6 * pollution + 0.4 * (1 + 2 * pollution) * log(Pv)
        phi_direct_atmosphere = transmitivity * phis_with_nebulosity * exp(-air_mass * Erayleigh * lLinke)
        return phi_direct_atmosphere, transmitivity

    def get_solar_gain(self, exposure_in_rad: float, slope_in_rad: float, datetime: Datetime, temperature: float = 15.4, humidity: float = .2, nebulosity_in_percentage: float = 0, pollution: float = 0.1):
        """
        compute the solar power on a 1m2 flat surface
        :param exposure_in_rad: angle of the surface with the north. O means north oriented, -pi/2 means West, pi/2 East and pi South oriented
        :param slope_in_rad: angle in radiants of the flat surface. 0 means horizontal directed to the sky zenith and pi/2 means vertical
        :param datetime: hour in the day
        :param temperature: outdoor temperature
        :param humidity: outdoor humidity
        :param nebulosity_in_percentage: cloudiness ie percentage of the sky covered by clouds
        :param pollution: pollution rate
        :return: phi_total, phi_direct_collected, phi_diffuse, phi_reflected
        """
        day_in_year, solartime_in_secondes = self.get_solar_time(datetime)
        altitude_in_rad, azimuth_in_rad, solarangle_in_rad, latitude_in_rad, declination_in_rad = self.get_solar_angles(day_in_year, solartime_in_secondes)
        phis = 1367 * (1 + 0.033 * cos(2 * pi * day_in_year / 365))
        phis_with_nebulosity = (1 - 0.75 * nebulosity_in_percentage ** 3.4) * phis
        phi_direct_atmosphere, transmitivity = self.get_solar_beam_transfer(phis_with_nebulosity, altitude_in_rad, temperature, humidity, pollution)
        incidence_in_rad = acos(cos(altitude_in_rad) * sin(slope_in_rad) * cos(azimuth_in_rad + exposure_in_rad) - sin(altitude_in_rad) * cos(slope_in_rad))
        if altitude_in_rad == 0 or (slope_in_rad !=0 and exposure_in_rad == 0):
           phi_direct_collected = 0
        else:
            phi_direct_collected = incidence_in_rad * phi_direct_atmosphere

        phi_diffuse = phis_with_nebulosity * (0.271 - 0.294 * transmitivity) * sin(altitude_in_rad)
        phi_reflected = self.albedo * phis_with_nebulosity * (0.271 + 0.706 * transmitivity) * sin(altitude_in_rad) * cos(slope_in_rad / 2) ** 2
        return phi_direct_collected + phi_diffuse + phi_reflected, phi_direct_collected, phi_diffuse, phi_reflected


if __name__ == '__main__':
    solar_gain = SolarGain()
    datestring = '27/07/2015'
    timestring = '13:00:00'
    slope = pi/2
    datetimestring = datestring + ' ' + timestring
    print('south: ', end='')
    print(solar_gain.get_solar_gain(SOUTH, pi / 2, common.timemg.stringdate_to_datetime(datetimestring)))
    print('east: ', end='')
    print(solar_gain.get_solar_gain(EAST, pi / 2, common.timemg.stringdate_to_datetime(datetimestring)))
    print('west: ', end='')
    print(solar_gain.get_solar_gain(WEST, pi / 2, common.timemg.stringdate_to_datetime(datetimestring)))
    print('north: ', end='')
    print(solar_gain.get_solar_gain(NORTH, pi / 2, common.timemg.stringdate_to_datetime(datetimestring)))
    exposure_in_rad_angles = [-pi + 2 * pi / 100 * i for i in range(100)]

    datetimes = [common.timemg.epochtimems_to_datetime(dt) for dt in range(common.timemg.stringdate_to_epochtimems(datestring + ' 0:00:00'), common.timemg.stringdate_to_epochtimems(datestring + ' 23:59:00'), 1000 * 60)]

    plt.subplot(2, 1, 1)
    iselect = 1
    phis_south = [solar_gain.get_solar_gain(SOUTH, slope, dt)[iselect] for dt in datetimes]
    phis_east = [solar_gain.get_solar_gain(EAST, slope, dt)[iselect] for dt in datetimes]
    phis_west = [solar_gain.get_solar_gain(WEST, slope, dt)[iselect] for dt in datetimes]
    phis_north = [solar_gain.get_solar_gain(NORTH, slope, dt)[iselect] for dt in datetimes]
    plt.plot(datetimes, phis_south, datetimes, phis_east, datetimes, phis_west, datetimes, phis_north)
    plt.legend(('south', 'east', 'west', 'north'))
    plt.axis('tight')
    plt.grid()

    plt.subplot(2, 1, 2)
    altitudes = [solar_gain.get_solar_angles(*solar_gain.get_solar_time(dt))[0] for dt in datetimes]
    azimuths = [solar_gain.get_solar_angles(*solar_gain.get_solar_time(dt))[1] for dt in datetimes]
    plt.plot(datetimes, altitudes, datetimes, azimuths)
    plt.legend(('altitude', 'azimuth'))
    plt.axis('tight')
    plt.grid()
    plt.show()
