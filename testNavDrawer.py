from kivy.uix.boxlayout import BoxLayout

from kivymd.app import MDApp
from kivy.lang import Builder

KV = '''
<TestNavigationDrawer>
    NavigationLayout:
        
    BoxLayout:
        orientation: 'vertical'
        MDToolbar:
            title: "Navigation Drawer"
            elevation: 10
            left_action_items: [['menu', lambda x: nav_drawer.toggle_nav_drawer()]]
            Widget:
    MDNavigationDrawer:
        id: nav_drawer
        ContentNavigationDrawer:
'''

class ContentNavigationDrawer(BoxLayout):
    pass
class TestNavigationDrawer(MDApp):
    def build(self):
        return Builder.load_string(KV)
        
TestNavigationDrawer().run()
