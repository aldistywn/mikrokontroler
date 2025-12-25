import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import numpy as np
import imclab  # Library buatan sendiri
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class PIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("iMCLab PID Controller Interface")
        self.root.geometry("1000x700")

        # --- VARIABEL PID DEFAULT ---
        self.kp = 1.5
        self.ki = 0.5
        self.kd = 0.05
        self.setpoint = 0.0
        self.running = False
        self.lab = None
        
        # Data Log untuk Grafik
        self.time_data = []
        self.sp_data = []
        self.rpm_data = []
        self.out_data = []
        self.start_time = 0
        self.window_size = 100 # Menampilkan 100 data terakhir

        # --- LAYOUT UI ---
        self.create_widgets()
        
        # Setup Protokol Tutup Aplikasi
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        # 1. FRAME KONTROL (Kiri)
        control_frame = ttk.LabelFrame(self.root, text="Kontrol & Parameter")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Tombol Koneksi
        self.btn_connect = ttk.Button(control_frame, text="🔌 Connect Arduino", command=self.connect_arduino)
        self.btn_connect.pack(pady=10, fill=tk.X)
        
        self.lbl_status = ttk.Label(control_frame, text="Status: Disconnected", foreground="red")
        self.lbl_status.pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Input Setpoint
        ttk.Label(control_frame, text="Setpoint (RPM):").pack(anchor='w')
        self.scale_sp = tk.Scale(control_frame, from_=0, to=10000, orient=tk.HORIZONTAL, command=self.update_params)
        self.scale_sp.set(0)
        self.scale_sp.pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Input PID
        ttk.Label(control_frame, text="Konstanta Kp:").pack(anchor='w')
        self.ent_kp = ttk.Entry(control_frame)
        self.ent_kp.insert(0, str(self.kp))
        self.ent_kp.pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="Konstanta Ki:").pack(anchor='w')
        self.ent_ki = ttk.Entry(control_frame)
        self.ent_ki.insert(0, str(self.ki))
        self.ent_ki.pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="Konstanta Kd:").pack(anchor='w')
        self.ent_kd = ttk.Entry(control_frame)
        self.ent_kd.insert(0, str(self.kd))
        self.ent_kd.pack(fill=tk.X, pady=2)

        ttk.Button(control_frame, text="Update PID Params", command=self.update_params).pack(pady=10, fill=tk.X)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Display Realtime
        self.lbl_rpm = ttk.Label(control_frame, text="RPM: 0", font=("Arial", 14, "bold"))
        self.lbl_rpm.pack(pady=5)
        
        self.lbl_out = ttk.Label(control_frame, text="Output: 0%", font=("Arial", 12))
        self.lbl_out.pack(pady=2)

        # 2. FRAME GRAFIK (Kanan)
        plot_frame = ttk.LabelFrame(self.root, text="Visualisasi Real-time")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Setup Matplotlib
        self.fig = Figure(figsize=(5, 5), dpi=100)
        
        # Subplot 1: Respon (RPM vs SP)
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Respon Sistem (RPM)")
        self.ax1.set_ylabel("RPM")
        self.line_sp, = self.ax1.plot([], [], 'g--', label='Setpoint')
        self.line_rpm, = self.ax1.plot([], [], 'r-', label='RPM Aktual')
        self.ax1.legend(loc='upper right')
        self.ax1.grid(True)

        # Subplot 2: Output (PWM)
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Usaha Kontrol (Output %)")
        self.ax2.set_ylabel("PWM (%)")
        self.ax2.set_ylim(0, 105)
        self.line_out, = self.ax2.plot([], [], 'b-', label='Output')
        self.ax2.fill_between([], [], color='blue', alpha=0.2)
        self.ax2.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def connect_arduino(self):
        try:
            self.lab = imclab.iMCLab()
            self.lbl_status.config(text="Status: Connected ✅", foreground="green")
            self.btn_connect.config(state=tk.DISABLED)
            
            # Mulai Thread PID
            self.running = True
            self.start_time = time.time()
            self.pid_thread = threading.Thread(target=self.pid_loop, daemon=True)
            self.pid_thread.start()
            
            # Mulai Animasi Grafik
            self.animate_plot()
            
        except Exception as e:
            messagebox.showerror("Error Koneksi", f"Gagal connect ke Arduino:\n{e}")

    def update_params(self, event=None):
        """Mengambil nilai dari input UI ke variabel program"""
        try:
            self.setpoint = float(self.scale_sp.get())
            self.kp = float(self.ent_kp.get())
            self.ki = float(self.ent_ki.get())
            self.kd = float(self.ent_kd.get())
        except ValueError:
            pass # Abaikan jika input kosong/huruf

    def pid_loop(self):
        """Looping PID yang berjalan di background thread"""
        prev_error = 0
        integral = 0
        last_time = time.time()

        while self.running:
            current_time = time.time()
            dt = current_time - last_time

            if dt >= 0.1: # Sampling 10Hz
                # 1. Baca RPM
                try:
                    current_rpm = self.lab.RPM
                except:
                    current_rpm = 0

                # 2. Hitung PID
                error = self.setpoint - current_rpm
                
                # P
                P = self.kp * error
                
                # I
                integral += error * dt
                integral = max(-100, min(100, integral)) # Anti-windup
                I = self.ki * integral
                
                # D
                D = self.kd * (error - prev_error) / dt if dt > 0 else 0
                
                # Total Output
                output = P + I + D
                output = max(0.0, min(100.0, output))
                
                # 3. Kirim ke Motor
                self.lab.op(output)

                # 4. Simpan Data untuk Grafik
                elapsed = current_time - self.start_time
                self.update_data_arrays(elapsed, self.setpoint, current_rpm, output)
                
                # 5. Update Label Teks UI (Gunakan after agar thread aman)
                self.root.after(0, self.update_labels, current_rpm, output)

                prev_error = error
                last_time = current_time
            
            time.sleep(0.01)

    def update_data_arrays(self, t, sp, rpm, out):
        """Menyimpan data dan menjaga agar array tidak terlalu panjang"""
        self.time_data.append(t)
        self.sp_data.append(sp)
        self.rpm_data.append(rpm)
        self.out_data.append(out)

        # Hapus data lama jika melebihi batas window
        if len(self.time_data) > self.window_size:
            self.time_data.pop(0)
            self.sp_data.pop(0)
            self.rpm_data.pop(0)
            self.out_data.pop(0)

    def update_labels(self, rpm, out):
        self.lbl_rpm.config(text=f"RPM: {rpm:.0f}")
        self.lbl_out.config(text=f"Output: {out:.1f}%")

    def animate_plot(self):
        """Fungsi update grafik periodik"""
        if not self.running:
            return

        # Update data garis
        self.line_sp.set_data(self.time_data, self.sp_data)
        self.line_rpm.set_data(self.time_data, self.rpm_data)
        self.line_out.set_data(self.time_data, self.out_data)

        # Rescale sumbu X dan Y otomatis
        if self.time_data:
            self.ax1.set_xlim(min(self.time_data), max(self.time_data) + 1)
            self.ax1.set_ylim(0, max(3000, max(self.sp_data), max(self.rpm_data)) + 100)
            
            self.ax2.set_xlim(min(self.time_data), max(self.time_data) + 1)
            # Y output tetap 0-100

        self.canvas.draw()
        
        # Update grafik setiap 500ms (0.5 detik) agar tidak berat
        self.root.after(500, self.animate_plot)

    def on_close(self):
        """Pembersihan saat aplikasi ditutup"""
        self.running = False
        if self.lab:
            try:
                self.lab.op(0) # Matikan motor
                self.lab.close()
            except:
                pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = PIDApp(root)
    root.mainloop()