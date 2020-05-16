from kivy.app import App
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen, ScreenManager

Builder.load_string("""
<LoginScreen>:
    BoxLayout:
        orientation:'vertical'
        Label:
            text: 'I am LoginScreen'
        Label:
            id: lbl1
        Button:
            text: 'Read'
            on_press: root.press_read()
        Button:
            text: 'Change'
            on_press:
                app.MY_NUMBER = app.MY_NUMBER + 1
                root.ids.lbl1.text = 'SharedVar is ' + str(app.MY_NUMBER)
        Button:
            text: 'Go to ScreenTwo'
            on_press: app.sm.current = "screen_2"
<MenuScreen>:
    BoxLayout:
        orientation:'vertical'
        Label:
            text: 'I am MenuScreen'
        Label:
            id: lbl2
        Button:
            text: 'Read'
            on_press: root.press_read()
        Button:
            text: 'Change'
            on_press:
                app.MY_NUMBER = app.MY_NUMBER + 1
                root.ids.lbl2.text = 'SharedVar is ' + str(app.MY_NUMBER)
        Button:
            text: 'Go to ScreenOne'
            on_press: app.sm.current = "screen_1"
""")

class LoginScreen(Screen):
   def press_read(self):
      app = App.get_running_app()
      self.ids.lbl1.text = "SharedVar is " + str(app.MY_NUMBER)

class MenuScreen(Screen):
   def press_read(self):
      app = App.get_running_app()
      self.ids.lbl2.text = "SharedVar is now " + str(app.MY_NUMBER)

class HandSetApp(App):
   MY_NUMBER = 0
   sm = ScreenManager()

   def build(self):
      HandSetApp.sm.add_widget(LoginScreen(name='screen_1'))
      HandSetApp.sm.add_widget(MenuScreen(name='screen_2'))
      return HandSetApp.sm

if __name__ == '__main__':
   HandSetApp().run()