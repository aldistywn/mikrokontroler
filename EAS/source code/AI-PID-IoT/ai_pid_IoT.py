import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import time
import threading
import numpy as np
import imclab
import joblib
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import paho.mqtt.client as mqtt
from scipy.interpolate import make_interp_spline
SMOOTH_GRAPH = True

class AIPIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("iMCLab AI-IoT Controller (MQTT Enabled)")
        self.root.geometry("1100x850") 
        self.last_mqtt_time = 0

        # --- KONFIGURASI MQTT ---
        self.mqtt_broker = "broker.hivemq.com"
        self.mqtt_port = 1883
        self.topic_prefix = "imclab/ai_iot"
        
        # Topik MQTT
        self.topic_pub_rpm = f"{self.topic_prefix}/rpm"
        self.topic_pub_sp  = f"{self.topic_prefix}/setpoint_monitor"
        self.topic_pub_pwm = f"{self.topic_prefix}/pwm"
        self.topic_sub_sp  = f"{self.topic_prefix}/setpoint_control"

        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message

        self.last_mqtt_time = 0
        
        # Mulai MQTT di background
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            print(f"📡 MQTT Connected to {self.mqtt_broker}")
        except Exception as e:
            print(f"⚠️ MQTT Gagal: {e}")

        # Parameter Default
        self.kp = 0.005; self.ki = 0.005; self.kd = 0.002
        self.setpoint = 0.0
        self.running = False
        self.lab = None
        self.ai_model = None
        self.use_ai = tk.BooleanVar(value=True) 

        # Variabel Internal
        self.integral = 0.0; self.prev_rpm = 0.0; self.rpm_filtered = 0.0 

        self.load_ai_model()
        
        # Data Logging
        self.time_data = []; self.sp_data = []; self.rpm_data = []; self.out_data = []
        self.window_size = 150 
        self.history_time = []; self.history_sp = []; self.history_rpm = []; self.history_out = []
        self.start_time = 0

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- FUNGSI MQTT ---
    def on_mqtt_connect(self, client, userdata, flags, rc):
        print(f"Terhubung ke Broker! Code: {rc}")
        # Subscribe ke topik kontrol
        client.subscribe(self.topic_sub_sp)

    def on_mqtt_message(self, client, userdata, msg):
        try:
            payload = str(msg.payload.decode("utf-8"))
            print(f"📩 Pesan dari HP: {payload}")
            
            # Update Setpoint
            new_sp = float(payload)
            self.setpoint = new_sp
            
            # Update UI Slider (biar sinkron)
            self.scale_sp.set(new_sp)
            
            # Trigger AI Tuning jika perlu
            if self.use_ai.get() and self.ai_model is not None and new_sp > 500:
                # Jalankan di thread aman agar tidak macet
                self.root.after(0, lambda: self.run_ai_tuning(new_sp))
                
        except Exception as e:
            print(f"❌ Error parsing MQTT: {e}")

    def load_ai_model(self):
        try:
            self.ai_model = joblib.load('pid_model.pkl')
            print("Otak AI Berhasil Dimuat!")
        except:
            print("Mode AI mati.")
            self.use_ai.set(False)

    def create_widgets(self):
        control_frame = ttk.LabelFrame(self.root, text="AI & Hybrid Control")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.btn_connect = ttk.Button(control_frame, text="🔌 Connect & Start", command=self.connect_arduino)
        self.btn_connect.pack(pady=10, fill=tk.X)
        self.lbl_status = ttk.Label(control_frame, text="Status: Disconnected", foreground="red")
        self.lbl_status.pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        chk_ai = ttk.Checkbutton(control_frame, text="Aktifkan AI Auto-Tuner", variable=self.use_ai, command=self.toggle_ai_inputs)
        chk_ai.pack(pady=5, anchor='w')
        
        ttk.Label(control_frame, text="Target RPM:").pack(anchor='w', pady=(10,0))
        self.scale_sp = tk.Scale(control_frame, from_=0, to=5000, orient=tk.HORIZONTAL) 
        self.scale_sp.bind("<ButtonRelease-1>", self.on_setpoint_change) 
        self.scale_sp.set(0)
        self.scale_sp.pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        self.frm_pid = ttk.Frame(control_frame)
        self.frm_pid.pack(fill=tk.X)
        ttk.Label(self.frm_pid, text="Kp:").pack(anchor='w')
        self.ent_kp = ttk.Entry(self.frm_pid); self.ent_kp.insert(0, str(self.kp)); self.ent_kp.pack(fill=tk.X)
        ttk.Label(self.frm_pid, text="Ki:").pack(anchor='w')
        self.ent_ki = ttk.Entry(self.frm_pid); self.ent_ki.insert(0, str(self.ki)); self.ent_ki.pack(fill=tk.X)
        ttk.Label(self.frm_pid, text="Kd:").pack(anchor='w')
        self.ent_kd = ttk.Entry(self.frm_pid); self.ent_kd.insert(0, str(self.kd)); self.ent_kd.pack(fill=tk.X)

        self.btn_update = ttk.Button(control_frame, text="Update Manual", command=self.manual_update_params)
        self.btn_update.pack(pady=10, fill=tk.X)

        self.lbl_ai_info = ttk.Label(control_frame, text="AI Info: Menunggu...", foreground="blue", wraplength=200)
        self.lbl_ai_info.pack(pady=10)
        self.lbl_floor_info = ttk.Label(control_frame, text="Safe Floor: 0%", foreground="green")
        self.lbl_floor_info.pack(pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        self.btn_capture = ttk.Button(control_frame, text="📸 Simpan Grafik Full", command=self.capture_full_graph)
        self.btn_capture.pack(pady=5, fill=tk.X)
        self.btn_reset = ttk.Button(control_frame, text="🔄 Reset Grafik & Data", command=self.reset_system)
        self.btn_reset.pack(pady=5, fill=tk.X)
        
        self.lbl_rpm = ttk.Label(control_frame, text="RPM: 0", font=("Arial", 16, "bold"))
        self.lbl_rpm.pack(pady=5)
        self.lbl_out = ttk.Label(control_frame, text="Power: 0%", font=("Arial", 12))
        self.lbl_out.pack(pady=2)
        self.lbl_mqtt = ttk.Label(control_frame, text="📡 MQTT: Online", foreground="blue")
        self.lbl_mqtt.pack(pady=2)

        plot_frame = ttk.LabelFrame(self.root, text="Real-time Performance")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("RPM vs Target (Smoothed)")
        self.line_sp, = self.ax1.plot([], [], 'g--', label='Target', linewidth=1.5)
        self.line_rpm, = self.ax1.plot([], [], 'r-', label='Actual', linewidth=2) 
        self.ax1.legend(loc='upper right'); self.ax1.grid(True, alpha=0.3)

        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Motor Power (%)")
        self.ax2.set_ylim(0, 105)
        self.line_out, = self.ax2.plot([], [], 'b-', linewidth=1.5)
        self.ax2.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def toggle_ai_inputs(self):
        state = tk.DISABLED if self.use_ai.get() else tk.NORMAL
        self.ent_kp.config(state=state); self.ent_ki.config(state=state); self.ent_kd.config(state=state); self.btn_update.config(state=state)

    def run_ai_tuning(self, target_rpm):
        if self.ai_model is None: return
        self.lbl_ai_info.config(text="AI Mencari Parameter...")
        self.root.update_idletasks()
        num_samples = 2000
        
        if target_rpm <= 2500: kp_range = (0.002, 0.007); ki_range = (0.002, 0.007); kd_range = (0.001, 0.003)
        elif target_rpm <= 3500: kp_range = (0.005, 0.007); ki_range = (0.005, 0.007); kd_range = (0.001, 0.003)
        elif target_rpm <= 4500: kp_range = (0.007, 0.009); ki_range = (0.007, 0.009); kd_range = (0.003, 0.007)
        else: kp_range = (0.008, 0.012); ki_range = (0.008, 0.010); kd_range = (0.002, 0.007)
            
        kp_c = np.random.uniform(kp_range[0], kp_range[1], num_samples)
        ki_c = np.random.uniform(ki_range[0], ki_range[1], num_samples)
        kd_c = np.random.uniform(kd_range[0], kd_range[1], num_samples)
        sp_arr = np.full(num_samples, target_rpm)
        input_data = pd.DataFrame({'setpoint': sp_arr, 'kp': kp_c, 'ki': ki_c, 'kd': kd_c})
        best_idx = np.argmin(self.ai_model.predict(input_data))
        self.kp = float(kp_c[best_idx]); self.ki = float(ki_c[best_idx]); self.kd = float(kd_c[best_idx])
        
        self.ent_kp.config(state=tk.NORMAL); self.ent_kp.delete(0, tk.END); self.ent_kp.insert(0, f"{self.kp:.4f}")
        self.ent_ki.config(state=tk.NORMAL); self.ent_ki.delete(0, tk.END); self.ent_ki.insert(0, f"{self.ki:.4f}")
        self.ent_kd.config(state=tk.NORMAL); self.ent_kd.delete(0, tk.END); self.ent_kd.insert(0, f"{self.kd:.4f}")
        if self.use_ai.get(): 
            self.ent_kp.config(state=tk.DISABLED); self.ent_ki.config(state=tk.DISABLED); self.ent_kd.config(state=tk.DISABLED)
        self.lbl_ai_info.config(text=f"✅ AI Configured for {target_rpm} RPM")

    def on_setpoint_change(self, event):
        val = float(self.scale_sp.get())
        self.setpoint = val
        if self.use_ai.get() and self.ai_model is not None and val > 500: self.run_ai_tuning(val)
        elif not self.use_ai.get(): self.manual_update_params()

    def manual_update_params(self):
        try: self.kp = float(self.ent_kp.get()); self.ki = float(self.ent_ki.get()); self.kd = float(self.ent_kd.get())
        except: pass

    def reset_system(self):
        self.integral = 0.0; self.prev_rpm = 0.0; self.rpm_filtered = 0.0
        self.time_data.clear(); self.sp_data.clear(); self.rpm_data.clear(); self.out_data.clear()
        self.history_time.clear(); self.history_sp.clear(); self.history_rpm.clear(); self.history_out.clear()
        self.start_time = time.time()
        self.line_sp.set_data([], []); self.line_rpm.set_data([], []); self.line_out.set_data([], [])
        self.canvas.draw()

    def connect_arduino(self):
        try:
            self.lab = imclab.iMCLab()
            self.lbl_status.config(text="Status: Connected ✅", foreground="green")
            self.btn_connect.config(state=tk.DISABLED)
            self.reset_system()
            self.running = True
            self.pid_thread = threading.Thread(target=self.pid_loop, daemon=True)
            self.pid_thread.start()
            self.animate_plot()
            self.toggle_ai_inputs()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal connect: {e}")

    def capture_full_graph(self):
        if len(self.history_time) < 10:
            messagebox.showwarning("Info", "Data belum cukup.")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
        if not filename: return
        try:
            fig_full = Figure(figsize=(12, 10), dpi=150)
            ax_top = fig_full.add_subplot(211)
            ax_top.plot(self.history_time, self.history_sp, 'g--', label='Target (SP)', linewidth=1.5)
            
            # Logic Smooth untuk Capture
            if SMOOTH_GRAPH and len(self.history_time) > 10:
                try:
                    x_smooth = np.linspace(min(self.history_time), max(self.history_time), 500)
                    spl = make_interp_spline(self.history_time, self.history_rpm, k=3)
                    y_smooth = spl(x_smooth)
                    ax_top.plot(x_smooth, y_smooth, 'r-', label='Actual RPM (Smooth)', linewidth=2)
                except:
                    ax_top.plot(self.history_time, self.history_rpm, 'r-', label='Actual RPM (Raw)', linewidth=1.5)
            else:
                ax_top.plot(self.history_time, self.history_rpm, 'r-', label='Actual RPM (Raw)', linewidth=1.5)

            ax_top.set_title(f"Full Response Analysis - RPM (Kp:{self.kp:.4f} Ki:{self.ki:.4f} Kd:{self.kd:.4f})")
            ax_top.set_ylabel("RPM"); ax_top.grid(True); ax_top.legend(loc='upper right')

            ax_bot = fig_full.add_subplot(212, sharex=ax_top) 
            ax_bot.plot(self.history_time, self.history_out, 'b-', label='PWM Output (%)', linewidth=1)
            ax_bot.set_title("Control Signal"); ax_bot.set_ylabel("Power (%)"); ax_bot.set_xlabel("Time (s)"); ax_bot.set_ylim(0, 105); ax_bot.grid(True); ax_bot.legend()
            fig_full.tight_layout(); fig_full.savefig(filename)
            messagebox.showinfo("Sukses", f"Grafik tersimpan: {filename}")
        except Exception as e:
            messagebox.showerror("Gagal", f"Error: {e}")

    def pid_loop(self):
        last_time = time.time()
        KICK_POWER = 83.0; RPM_ALIVE = 500

        while self.running:
            current_time = time.time()
            dt = current_time - last_time

            if dt >= 0.1: 
                try: raw_rpm = self.lab.RPM
                except: raw_rpm = 0

                self.rpm_filtered = (0.7 * self.rpm_filtered) + (0.3 * raw_rpm)
                pv = self.rpm_filtered
                error = self.setpoint - pv
                P = self.kp * error
                potential_integral = self.integral + (error * dt)
                d_rpm = (pv - self.prev_rpm) / dt if dt > 0 else 0
                D = -self.kd * d_rpm
                
                # Dynamic Floor
                if self.setpoint < 2200: current_floor = 65.0
                elif self.setpoint < 3500: current_floor = 60.0
                elif self.setpoint < 4500: current_floor = 57.0
                else: current_floor = 55.0
                self.root.after(0, lambda: self.lbl_floor_info.config(text=f"Active Floor: {current_floor}%"))

                op_temp = P + (self.ki * potential_integral) + D
                if op_temp > 100.0 or (op_temp < current_floor and error < 0): pass
                else: self.integral = potential_integral
                I = self.ki * self.integral
                op = P + I + D
                
                if pv < RPM_ALIVE and op > 1.0: op = max(op, KICK_POWER) 
                else:
                    if op > 1.0: op = max(op, current_floor)

                op = max(0.0, min(100.0, op))
                self.lab.op(op)
                self.prev_rpm = pv
                last_time = current_time

                elapsed = current_time - self.start_time
                self.time_data.append(elapsed); self.sp_data.append(self.setpoint)
                self.rpm_data.append(pv); self.out_data.append(op)
                if len(self.time_data) > self.window_size:
                    self.time_data.pop(0); self.sp_data.pop(0); self.rpm_data.pop(0); self.out_data.pop(0)

                self.history_time.append(elapsed); self.history_sp.append(self.setpoint)
                self.history_rpm.append(pv); self.history_out.append(op)
                
                # --- KIRIM DATA KE MQTT ---
                if (current_time - self.last_mqtt_time) > 0.5:
                    try:
                        self.mqtt_client.publish(self.topic_pub_rpm, f"{pv:.1f}")
                        self.mqtt_client.publish(self.topic_pub_pwm, f"{op:.1f}")
                        self.mqtt_client.publish(self.topic_pub_sp, f"{self.setpoint:.0f}")
                        self.last_mqtt_time = current_time 
                    except: pass

                self.root.after(0, self.update_labels, raw_rpm, op)
            
            time.sleep(0.01)

    def update_labels(self, rpm, out):
        self.lbl_rpm.config(text=f"RPM: {rpm:.0f}")
        self.lbl_out.config(text=f"Power: {out:.1f}%")

    def animate_plot(self):
        if not self.running: return
        self.line_sp.set_data(self.time_data, self.sp_data)
        
        if SMOOTH_GRAPH and len(self.time_data) > 4:
            try:
                x_new = np.linspace(min(self.time_data), max(self.time_data), 300)
                spl = make_interp_spline(self.time_data, self.rpm_data, k=3) 
                y_smooth = spl(x_new); self.line_rpm.set_data(x_new, y_smooth)
            except: self.line_rpm.set_data(self.time_data, self.rpm_data)
        else: self.line_rpm.set_data(self.time_data, self.rpm_data)
            
        self.line_out.set_data(self.time_data, self.out_data)
        if self.time_data:
            self.ax1.set_xlim(min(self.time_data), max(self.time_data) + 0.5)
            max_y = max(5500, max(self.sp_data), max(self.rpm_data)); self.ax1.set_ylim(0, max_y + 200)
            self.ax2.set_xlim(min(self.time_data), max(self.time_data) + 0.5)
        self.canvas.draw(); self.root.after(200, self.animate_plot)

    def on_close(self):
        self.running = False
        try: self.mqtt_client.loop_stop() # Stop MQTT
        except: pass
        if self.lab:
            try: self.lab.op(0); self.lab.close()
            except: pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AIPIDApp(root)
    root.mainloop()