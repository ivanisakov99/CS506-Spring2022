import tkinter
from PIL import Image, ImageDraw
import io
import PySimpleGUI as sg

def draw_gym():
    img = Image.open('./leisure/gym.jpg')
    img.thumbnail((1200, 800))
    bio = io.BytesIO()
    img.save(bio, format='PNG')

    layout = [[[sg.Text('Gym:', font=('Arial', 25)),],
        [sg.Image(data=bio.getvalue())]],
        [sg.Button('Dont show me anymore')]]

    window = sg.Window('Image', layout, size=(1200, 800))

    while True:
        event, values = window.read()
        if event == 'Dont show me anymore' or event == sg.WIN_CLOSED:
            break

    return
