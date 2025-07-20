import ipywidgets as widgets
from IPython.display import display

topic = widgets.Text(
     description = 'Tema:',
     placeholder='Ex: Saúde mental, Alimentação saudável, prevenção, etc...',
     layout=widgets.Layout(width='500px')
 )

display(topic)
f