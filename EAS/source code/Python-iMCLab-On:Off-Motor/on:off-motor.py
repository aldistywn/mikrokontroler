import tkinter as tk
import serial

arduino = serial.Serial('/dev/cu.usbserial-0001', 115200, timeout=0)

def send_data(value):
  arduino.write(bytes([int(value)]))

window = tk.Tk()
window.title("Kontrol Simpel")

on_button = tk.Button(window, text="ON", command=lambda: send_data(200))
on_button.pack(pady=5, padx=20, fill=tk.X)

off_button = tk.Button(window, text="OFF", command=lambda: send_data(0))
off_button.pack(pady=5, padx=20, fill=tk.X)

slider = tk.Scale(
    window,
    from_=0,
    to=200,
    orient=tk.HORIZONTAL,
    command=send_data
)
slider.set(128)
slider.pack(pady=10, padx=20)

window.mainloop()