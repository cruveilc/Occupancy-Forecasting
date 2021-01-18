from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.widget import Widget
import matplotlib.pyplot as plt
from kivy.uix.button import Button

plt.plot([1, 23, 2, 4])
plt.ylabel('some numbers')

class MyApp(App):

    def build(self):
        box = BoxLayout()
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        box.add_widget(FigureCanvasKivyAgg(plt.gcf()))
        box.add_widget(Button())
        box.add_widget(Button())




        return box

MyApp().run()