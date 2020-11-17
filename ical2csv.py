from icalendar import Calendar, Event
from datetime import datetime
import dateparser

g = open('stephane_stephane.ploix@gmail.com.ics','rb')
gcal = Calendar.from_ical(g.read().decode())
for component in gcal.walk():
    if component.name == "VEVENT":

        print(component.get('summary'))
        print(component.get('dtstart').dt)
        if component.get('dtend') is not None:

            print(component.get('dtend').dt)
        else:
            print("error")

        print(component.get('dtstamp').dt)

g.close()

