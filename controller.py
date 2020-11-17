from icalendar import Calendar, Event
from datetime import datetime, date, timedelta
import json
import csv
import os
# from numpy import random
from random import randint
#from occupantSimulator import bayesianOccupantSimulator
__all__ = ['OccupantTimeStepper', 'Icalendar']

__relative_path__ = '../occupants/'


class CalendarEvent():

    def __init__(self, vevent_entry):
        self.start_datetime = vevent_entry.decoded('DTSTART')
        if type(self.start_datetime) == date:
            self.start_datetime = datetime.combine(self.start_datetime, datetime.min.time()).replace(tzinfo=None)
        else:
            self.start_datetime = self.start_datetime.replace(tzinfo=None)
        self.end_datetime = vevent_entry.decoded('DTEND')
        if type(self.end_datetime) == date:
            self.end_datetime = datetime.combine(self.end_datetime, datetime.max.time()).replace(tzinfo=None)
        else:
            self.end_datetime = self.end_datetime.replace(tzinfo=None)
        self.summary = vevent_entry.get('SUMMARY')
        if self.summary is not None:
            self.summary = self.summary.encode('utf-8').decode('utf-8', 'replace')
        self.location = vevent_entry.get('LOCATION')
        if self.location is not None:
            self.location = self.location.encode('utf-8').decode('utf-8', 'replace')

    @property
    def duration_in_minutes(self):
        return int((self.end_datetime - self.start_datetime).total_seconds()/60)

    @property
    def start_date(self):
        return self.start_datetime.date()

    @property
    def end_date(self):
        return self.end_datetime.date()

    @property
    def start_time(self):
        return self.start_datetime.time()

    @property
    def end_time(self):
        return self.end_datetime.time()

    @property
    def time_interval(self):
        return TimeInterval(self.start_datetime, self.end_datetime)

    def __lt__(self, other):
        return self.start_datetime.replace(tzinfo=None).__lt__(other.start_datetime.replace(tzinfo=None))

    def __str__(self):
        return '%s > %s (duration %i min): %s @ %s' % (self.start_datetime, self.end_datetime, self.duration_in_minutes, self.summary, self.location)


class TimeInterval():

    def __init__(self, start_datetime, end_datetime):
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

    @property
    def width(self):
        return self.end_datetime - self.start_datetime

    def subtract(self, other_time_interval):
        if (other_time_interval.start_datetime > self.end_datetime) or (other_time_interval.end_datetime < self.start_datetime):
            return [TimeInterval(self.start_datetime, self.end_datetime)]
        elif (other_time_interval.start_datetime < self.start_datetime) and (other_time_interval.end_datetime > self.start_datetime):
            return [TimeInterval(min(max(self.start_datetime, other_time_interval.end_datetime),self.end_datetime), self.end_datetime)]
        elif (other_time_interval.start_datetime < self.end_datetime) and (other_time_interval.end_datetime > self.end_datetime):
            return [TimeInterval(self.start_datetime, max(self.start_datetime, min(self.end_datetime, other_time_interval.start_datetime)))]
        else:
            return [TimeInterval(self.start_datetime,other_time_interval.start_datetime), TimeInterval(other_time_interval.end_datetime, self.end_datetime)]

    def intersect(self, other_time_interval):
        start_datetime = max(self.start_datetime, other_time_interval.start_datetime)
        end_datetime = min(self.end_datetime, other_time_interval.end_datetime)
        if end_datetime < start_datetime:
            return None
        else:
            return TimeInterval(start_datetime, end_datetime)

    def __str__(self):
        return '[%s, %s]' % (self.start_datetime, self.end_datetime)

class Icalendar():

    @staticmethod
    def string_to_datetime(string_date: str):
        try:
            _datetime = datetime.strptime(string_date, '%d/%m/%Y %H:%M')
        except ValueError:
            _datetime = datetime.strptime(string_date + ' 0:0', '%d/%m/%Y %H:%M')
        return _datetime

    def __init__(self, ics_file_name: str, start_string_date: '%d/%m/%Y %H:%M'=None):
        calendar_file = open(__relative_path__ + ics_file_name, 'rb')
        if start_string_date is not None:
            self.start_date = Icalendar.string_to_datetime(start_string_date)
        else:
            self.start_date = None
        ical_data = Calendar.from_ical(calendar_file.read())
        self.events = []
        for ical_entry in ical_data.walk():
            if ical_entry.name == 'VEVENT':
                event = CalendarEvent(ical_entry)
                if self.start_date is not None and event.start_datetime >= self.start_date:
                    self.events.append(event)
        self.events = sorted(self.events)
        calendar_file.close()

    def crossing(self, studied_time_interval: 'TimeInterval'):
        intersecting_events = list()
        for event in self.events:
            if (event.start_datetime < studied_time_interval.end_datetime) and (event.end_datetime > studied_time_interval.start_datetime):
                intersecting_events.append(event)
        return intersecting_events

    def no_booked_timedelta(self, studied_time_interval: 'TimeInterval'):
        remaining_time_intervals = [studied_time_interval]
        for intersecting_event in self.crossing(studied_time_interval):
            new_remaining_time_intervals = list()
            for remaining_time_interval in remaining_time_intervals:
                resulting_time_intervals = remaining_time_interval.subtract(intersecting_event.time_interval)
                if len(resulting_time_intervals) > 0:
                    new_remaining_time_intervals.extend(resulting_time_intervals)
            remaining_time_intervals = new_remaining_time_intervals
        remaining_width = timedelta(0)
        for remaining_time_interval in remaining_time_intervals:
            remaining_width = remaining_width + remaining_time_interval.width
        return remaining_width

    def __str__(self):
        string = ''
        for event in self.events:
            string += event.__str__() + '\n'
        return string


class OccupantTimeStepper():

    DAYS_OF_WEEK = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    def __init__(self, start_string_date: '%d/%m/%Y %H:%M', time_step_in_minutes: int=60, vacation_calendar_name: str=None, busy_calendar_name: str=None):
        self.start_date = Icalendar.string_to_datetime(start_string_date)
        if vacation_calendar_name is not None:
            self.vacation_calendar = Icalendar(vacation_calendar_name, start_string_date)
        else:
            self.vacation_calendar = None
        if busy_calendar_name is not None:
            self.busy_calendar = Icalendar(busy_calendar_name, start_string_date)
        else:
            self.busy_calendar = None
        self.current_datetime = None
        self.time_step = timedelta(minutes=time_step_in_minutes)
        self.vacation_days_of_week = list()
        self.working_time_periods = list()
        self.specific_time_periods = list()

    def reset(self):
        self.current_datetime = None

    def set_vacation_days_of_week(self, *datetime_days_of_week_in_string: tuple):
        self.vacation_days_of_week = list()
        for datetime_day_of_week_in_string in datetime_days_of_week_in_string:
            self.vacation_days_of_week.append(OccupantTimeStepper.DAYS_OF_WEEK[datetime_day_of_week_in_string])

    def set_working_time_periods(self, *working_periods: list):
        self.working_time_periods = list()
        for working_period in working_periods:
            self.working_time_periods.append((datetime.strptime(working_period[0], '%H:%M').time(), datetime.strptime(working_period[1], '%H:%M').time()))

    def set_specific_time_periods(self, *specific_periods: list):
        self.specific_time_periods = list()
        if specific_periods is not None:
            for specific_period in specific_periods:
                self.specific_time_periods.append((datetime.strptime(specific_period[0], '%H:%M').time(), datetime.strptime(specific_period[1], '%H:%M').time()))

    def step(self):
        if self.current_datetime is None:
            self.current_datetime = self.start_date
        else:
            self.current_datetime = self.current_datetime + self.time_step

    @property
    def day_of_week(self):
        weekday_number = self.current_datetime.weekday()
        return weekday_number, list(OccupantTimeStepper.DAYS_OF_WEEK.keys())[list(OccupantTimeStepper.DAYS_OF_WEEK.values()).index(weekday_number)]

    @property
    def is_weekend(self):
        if self.current_datetime.weekday() in self.vacation_days_of_week:
            return True
        else:
            return False

    @property
    def working_ratio(self):
        if self.is_weekend:
            return 0
        current_time_interval = TimeInterval(self.current_datetime, self.current_datetime + self.time_step)
        effective_working_time_interval = None
        for working_time_period in self.working_time_periods:
            working_time_interval = TimeInterval(datetime.combine(self.current_datetime, working_time_period[0]),datetime.combine(self.current_datetime, working_time_period[1]))
            intersected_time_interval = current_time_interval.intersect(working_time_interval)
            if intersected_time_interval is not None:
                effective_working_time_interval = intersected_time_interval
        if effective_working_time_interval is None or effective_working_time_interval.width == timedelta(0):
            return 0
        else:
            if self.vacation_calendar is None:
                return effective_working_time_interval.width / current_time_interval.width
            else:
                return self.vacation_calendar.no_booked_timedelta(effective_working_time_interval) / current_time_interval.width

    @property
    def busy_ratio(self):
        current_time_interval = TimeInterval(self.current_datetime, self.current_datetime + self.time_step)
        if self.busy_calendar is None:
            return 0
        else:
            return 1 - self.busy_calendar.no_booked_timedelta(current_time_interval) / current_time_interval.width

    @property
    def specific_ratios(self):
        if len(self.specific_time_periods) == 0:
            return []
        else:
            ratios = []
            current_time_interval = TimeInterval(self.current_datetime, self.current_datetime + self.time_step)
            for i in range(len(self.specific_time_periods)):
                specific_time_period = TimeInterval(datetime.combine(self.current_datetime, self.specific_time_periods[i][0]), datetime.combine(self.current_datetime, self.specific_time_periods[i][1]))
                intersection = current_time_interval.intersect(specific_time_period)
                if intersection is None:
                    ratios.append(0)
                else:
                    ratios.append(intersection.width / current_time_interval.width)
            return ratios


class TimeState():
    def __init__(self, config_occupants):
        self.variables = ['datetime', 'weekday', 'weekend']
        self.occupants=[]
        for occupant in config_occupants:
            occupant_name = occupant['name']
            self.occupants.append(occupant_name)
            occupant_variables = ['working_%s' % occupant_name, 'busy_%s' % occupant_name]
            specific_time_periods = occupant['specific_time_periods']
            for i in range(len(specific_time_periods)):
                occupant_variables.append('period%i_%s' % (i, occupant_name))
            self.variables.extend(occupant_variables)
        self.values = dict()


    def set_variable_value(self, variable_name: str, variable_value):
        if variable_name in self.variables:
            self.values[variable_name] = variable_value
        else:
            raise Exception('Variable %s is not in state' % variable_name)

    def set_occupant_variable_value(self, occupant_name: str, occupant_variable_name: str, variable_value):
        self.set_variable_value(occupant_variable_name + '_' + occupant_name, variable_value)

    def set_occupant_specific_variables_values(self, occupant_name: str, variable_value_list: list):
        for i in range(len(variable_value_list)):
            self.set_occupant_variable_value(occupant_name, 'period%i' % i , variable_value_list[i])

    def values_list(self):
        variable_values = []
        for variable in self.variables:
            if variable in self.values:
                variable_values.append(self.values[variable])
            else:
                variable_values.append('')
        return variable_values

    def __str__(self):
        __str = 'TIME_STATE: '
        values = self.values_list()
        for i in range(len(self.variables)):
            __str += '%s=%s, ' % (self.variables[i], values[i].__str__())
        return __str


class Actions():

    def __init__(self):
        self.actions = dict()
        self.actions['Occupancy'] = 1.7 #changeable
        self.actions['RadiativeGain'] = 30 #changeable
        self.actions['ConvectiveGain'] = 70
        self.actions['HumidityProduction'] = 12
        self.actions['TemperatureSetPointLevel'] = 2
        self.actions['HeatingSystemSwitch'] = False
        self.actions['MobileHeaterSwitch'] = False
        self.actions['MobileHeaterTemperatureSetPointLevel'] = 4
        self.actions['AverageWindowsOpening'] = 0.3 #changeable
        self.actions['AverageShutterClosing'] = 0.2
        self.actions['AverageDoorOpening'] = 0.7 #changeable

    def build_actions(self):
        H358entrance_door = dict()
        H358entrance_door['DoorID'] = 'H358entrance'
        H358entrance_door['AverageOpening'] = self.actions['AverageDoorOpening']
        LeftSide_window = dict()
        LeftSide_window['WindowID'] = 'LeftSideWindow'
        LeftSide_window['AverageWindowsOpening'] = self.actions['AverageWindowsOpening']
        LeftSide_window['AverageShutterClosing'] = self.actions['AverageShutterClosing']
        MobileHeater_appliance = dict()
        MobileHeater_appliance['ApplianceID'] = 'MobileHeater'
        MobileHeater_appliance['MobileHeaterSwitch'] = self.actions['MobileHeaterSwitch']
        MobileHeater_appliance['MobileHeaterTemperatureSetPointLevel'] = self.actions['MobileHeaterTemperatureSetPointLevel']
        H358_room = dict()
        H358_room['RoomID'] = 'H358'
        # H358_room['Presence'] = (self.actions['Occupancy'] > 0.2)
        H358_room['Occupancy'] = (self.actions['Occupancy'])
        H358_room['RadiativeGain'] = self.actions['RadiativeGain']
        H358_room['ConvectiveGain'] = self.actions['ConvectiveGain']
        H358_room['HumidityProduction'] = self.actions['HumidityProduction']
        H358_room['CO2Production'] = 7 * self.actions['Occupancy']
        H358_room['TemperatureSetPointLevel'] = self.actions['TemperatureSetPointLevel']
        H358_room['HeatingSystemSwitch'] = self.actions['HeatingSystemSwitch']
        H358_room['Appliances'] = [MobileHeater_appliance]
        H358_room['Windows'] = [LeftSide_window]
        H35x_zone = dict()
        H35x_zone['ZoneName'] = 'H35x'
        H35x_zone['Rooms'] = [H358_room]
        H35x_zone['Doors'] = [H358entrance_door]
        message = dict()
        message['BuildingID'] = 'INPG'
        message['Zones'] = [H35x_zone]
        return json.dumps(message)

    def __str__(self):
        __str = 'ACTIONS: '
        for action_name in self.actions:
            __str += '%s=%s, ' % (action_name, self.actions[action_name].__str__())
        return __str

class PhysicalState():

    def __init__(self, state: dict):
        self.values_dict = state

    def __str__(self):
        return 'PHYSICAL_STATE: ' + self.values_dict.__str__()

    def default(self):
        return self.occupant_actions.build_actions()

    #
    # class SimulationController():
    #
    #     def __init__(self, configuration_filename: str='setup.json'):
    #         json_config = open(__relative_path__ + configuration_filename).read()
    #         config = json.loads(json_config)
    #         self.start_string_date = config['General']['start_string_date']
    #         self.time_step_in_minutes = int(config['General']['time_step_in_minutes'])
    #         self.results_filename = config['General']['results_filename']
    #         self.log = config['General']['log']
    #         self.current_datetime = self.start_string_date
    #         self.occupant_time_steppers_dict = dict()
    #         self.occupant_actions = Actions()
    #         for occupant in config['Occupants']:
    #             occupant_name = occupant['name']
    #             vacation_calendar = occupant['vacation_calendar']
    #             if vacation_calendar is not None and vacation_calendar == '':
    #                 vacation_calendar = None
    #             busy_calendar = occupant['busy_calendar']
    #             if busy_calendar is not None and busy_calendar == '':
    #                 busy_calendar = None
    #             vacation_days_of_week = occupant['vacation_days_of_week']
    #             working_time_periods = occupant['working_time_periods']
    #             specific_time_periods = occupant['specific_time_periods']
    #             self.occupant_time_steppers_dict[occupant_name] = OccupantTimeStepper(self.start_string_date, self.time_step_in_minutes, vacation_calendar, busy_calendar)
    #             self.occupant_time_steppers_dict[occupant_name].set_vacation_days_of_week(*vacation_days_of_week)
    #             self.occupant_time_steppers_dict[occupant_name].set_working_time_periods(*working_time_periods)
    #             self.occupant_time_steppers_dict[occupant_name].set_specific_time_periods(*specific_time_periods)
    #         self.time_state = TimeState(config['Occupants'])
    #         self.csv_file = None
    #         self.csv_writer = None
    #         if self.results_filename != '':
    #             self.csv_file = open(__relative_path__ + 'temp.csv', 'w', encoding='utf-8', newline='\n')
    #             self.csv_writer = csv.writer(self.csv_file, dialect=csv.excel, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    #             self.csv_writer.writerow(self.time_state.variables)
    #             self.csv_file.flush()
    #
    # def step(self, physical_state: dict):
    #     for occupant_name in self.occupant_time_steppers_dict:
    #         self.occupant_time_steppers_dict[occupant_name].step()
    #         self.time_state.set_variable_value('datetime', self.occupant_time_steppers_dict[occupant_name].current_datetime)
    #         self.time_state.set_variable_value('weekday', self.occupant_time_steppers_dict[occupant_name].day_of_week[1])
    #         self.time_state.set_variable_value('weekend', self.occupant_time_steppers_dict[occupant_name].is_weekend)
    #         self.time_state.set_occupant_variable_value(occupant_name, 'working', self.occupant_time_steppers_dict[occupant_name].working_ratio)
    #         self.time_state.set_occupant_variable_value(occupant_name, 'busy', self.occupant_time_steppers_dict[occupant_name].busy_ratio)
    #         self.time_state.set_occupant_specific_variables_values(occupant_name, self.occupant_time_steppers_dict[occupant_name].specific_ratios)
    #     self.occupant_actions = self.call_bayesian_simulator_step(self.time_state, PhysicalState(physical_state), self.occupant_actions)
    #     if self.log:
    #         pass
    #     if self.csv_file is not None:
    #         self.csv_writer.writerow(self.time_state.values_list())
    #         self.csv_file.flush()
    #     return self.occupant_actions.build_actions()
    #
    # def call_bayesian_simulator_step(self, time_state: 'occupants.controller.TimeState', physical_state: 'PhysicalState', occupant_actions: 'occupants.controller.Actions'):
    #     (occupantBusyState,weather)=self.extractDataFromTimeState(time_state)
    #     simulator=bayesianOccupantSimulator(occupantBusyState,weather,dict(ProfessorPresence=''),None)
    #     ProfessorPresence=simulator.perform()
    #     # print("Prof:",presence1)
    #     simulator=bayesianOccupantSimulator(occupantBusyState,weather,dict(PermanentPresence=''),None)
    #     PermanentPresence=simulator.perform()
    #     # print("Perm:",presence2)
    #     simulator=bayesianOccupantSimulator(occupantBusyState,weather,dict(IntermittentPresence=''),None)
    #     IntermittentPresence=simulator.perform()
    #     # print("Inter:",presence3)
    #     simulator=bayesianOccupantSimulator(occupantBusyState,weather,dict(VisitorPresence=''),None)
    #     VisitorPresence=simulator.perform()
    #     # print("Vist:",presence4)
    #     simulator=bayesianOccupantSimulator(occupantBusyState,weather,dict(GuestPresence=''),None)
    #     GuestPresence=simulator.perform()
    #     # print("Guest:",presence5)
    #     # print("Presence :",presence1+presence2+presence3+presence4+presence5)
    #     occupancy = ProfessorPresence+PermanentPresence+IntermittentPresence+VisitorPresence+GuestPresence
    #     simulator = bayesianOccupantSimulator(occupantBusyState,weather,dict(DoorMovement=''),'PastDoorMovement')
    #     doorStateAtEachStep = simulator.perform()
    #     # print("door : ",doorStateAtEachStep)
    #     if(doorStateAtEachStep =="close"):
    #         occupant_actions.actions["AverageDoorOpening"] = 0.1
    #     elif(doorStateAtEachStep =="move"):
    #         occupant_actions.actions["AverageDoorOpening"] = 0.5
    #         occupant_actions.actions["Occupancy"] = occupancy
    #     else:
    #         occupant_actions.actions["AverageDoorOpening"] = 0.9
    #         occupant_actions.actions["Occupancy"] = occupancy
    #     simulator = bayesianOccupantSimulator(occupantBusyState,weather,dict(WindowMovement=''),'PastWindowMovement')
    #     windowStateAtEachStep = simulator.perform()
    #     # print("window : ",windowStateAtEachStep)
    #     if(doorStateAtEachStep =="close"):
    #         occupant_actions.actions["AverageWindowsOpening"] = 0.1
    #         occupant_actions.actions["Occupancy"] = occupancy
    #     elif(doorStateAtEachStep =="move"):
    #         occupant_actions.actions["AverageWindowsOpening"] = 0.5
    #         occupant_actions.actions["Occupancy"] = occupancy
    #     else:
    #         occupant_actions.actions["AverageWindowsOpening"] = 0.9
    #         occupant_actions.actions["Occupancy"] = occupancy
    #     # print(physical_state)
    #     #TODO call the Bayesian Network simulator and update occupant_actions
    #     # print("occupant_actions : ", occupant_actions)
    #     return occupant_actions
    #
    # def close(self):
    #     if self.csv_file is not None:
    #         self.csv_file.close()
    #         os.rename(__relative_path__ + 'temp.csv', __relative_path__ + self.results_filename)
    #
    # def extractDataFromTimeState(self,time_state):
    #     occupantBusyState = dict()
    #     values = time_state.values
    #     # print("values",values)
    #     occupantList = time_state.occupants
    #     for occupant in occupantList:
    #         state = self.determineOccupantBusyState(values['weekend'],values['working_%s'%occupant],values['busy_%s'%occupant],occupant)
    #         occupantBusyState[occupant]=state
    #     weather = self.determineWeather(values['datetime'].month)
    #     return occupantBusyState,weather
    # def determineWeather(self,datetime):
    #     if datetime<=4:
    #         return 'Cold'
    #     elif datetime<=8:
    #         return 'Hot'
    # def determineOccupantBusyState(self,weekend,workingState,busyState,occupant):
    #     # print("weekend ",weekend,"workingState",workingState,"busyState",busyState,occupant)
    #     if weekend == True or workingState == 0:
    #         return 'outOfWorkingTime'
    #     else:
    #         if workingState == 0.5:
    #             return 'Intermediate'
    #         elif workingState == 1:
    #             return 'Present'
    #         else:
    #             return 'Absent'
    #         if str(occupant) =='stephane':
    #             if weekend == False and workingState == 1:
    #                 if busyState == 1:
    #                     return 'Present'
    #                 elif workingState == 0.5:
    #                     return 'Intermediate'
    #                 else:
    #                     return 'Absent'
    #         else:
    #             if weekend == False and workingState == 1:
    #                 if busyState == 1:
    #                     return 'Present'
    #                 elif workingState == 0.5:
    #                     return 'Intermediate'
    #                 else:
    #                     return 'Absent'


if __name__ == '__main__':
    __relative_path__ = ''
    controller = Icalendar('stephane_stephane.ploix@gmail.com.ics')
    # for iteration in range(10*24):
    #     controller.step({})
    # controller.close()