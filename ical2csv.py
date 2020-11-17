from icalendar import Calendar, Event
from datetime import datetime

import pandas as pd
import numpy as np


evenement = []
debut = []
fin =[]


g = open('stephane_stephane.ploix@gmail.com.ics','rb')
gcal = Calendar.from_ical(g.read().decode())
for component in gcal.walk():
    if component.name == "VEVENT":

        print(component.get('summary'))
        evenement.append(str((component.get('summary'))))
        print(component.get('dtstart').dt)
        debut.append(component.get('dtstart').dt)
        if component.get('dtend') is not None:

            print(component.get('dtend').dt)
            fin.append(component.get('dtend').dt)
        else:
            print("error")
            fin.append("Nan")

        print(component.get('dtstamp').dt)

g.close()

calendrier = pd.DataFrame({'evenement': evenement,'debut':debut,'fin':fin})

print(type(calendrier['debut'][0]))


#calendrier.sort('debut')

print(calendrier[0:50])