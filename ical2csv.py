from icalendar import Calendar, Event
from datetime import datetime

import pandas as pd
import numpy as np

from datetime import datetime

date_str ="2007-04-30 13:30:00+00:00"
date_str= date_str[:-6]
print(date_str)
date_obj = datetime.strptime(date_str,'%Y-%m-%d %H:%M:%S')
print(date_obj)

evenement = []
debut = []
fin =[]


g = open('stephane_stephane.ploix@gmail.com.ics','rb')
gcal = Calendar.from_ical(g.read().decode())
for component in gcal.walk():
    if component.name == "VEVENT":

        evenement.append(str((component.get('summary'))))
        if len(str(component.get('dtstart').dt)) >12:
            debut.append(datetime.strptime(str(component.get('dtstart').dt)[:-6],'%Y-%m-%d %H:%M:%S'))
        else:
            debut.append(datetime.strptime(str(component.get('dtstart').dt), '%Y-%m-%d'))
        if component.get('dtend') is not None:
            fin.append(component.get('dtend').dt)
        else:
            fin.append("Nan")

g.close()

calendrier = pd.DataFrame({'evenement': evenement,'debut':debut,'fin':fin},index=)

calendrier['debut'] =pd.to_datetime(calendrier.debut)

calendrier.sort_values(['debut'], inplace=True)
print(calendrier)


