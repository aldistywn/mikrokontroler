import time
import csv
import os
import numpy as np
import imclab
from datetime import datetime

class SmartTieredCollector:
    def __init__(self):
        self.lab = None
        self.training_data = []
        self.experiment_count = 0
        self.rpm_filtered = 0.0
        
    def connect(self):
        try:
            self.lab = imclab.iMCLab()
            print("✅ Terhubung ke iMCLab")
            return True
        except Exception as e:
            print(f"❌ Gagal connect: {e}")
            return False

    # Fungsi untuk menentukan Range Pencarian berdasarkan Target RPM
    def get_search_range(self, target_rpm):
        if target_rpm <= 2500:
            return {
                'kp_min': 0.002, 'kp_max': 0.007,
                'ki_min': 0.002, 'ki_max': 0.007,
                'kd_min': 0.001, 'kd_max': 0.003
            }
        elif target_rpm <= 3500:
            return {
                'kp_min': 0.005, 'kp_max': 0.007,
                'ki_min': 0.005, 'ki_max': 0.007,
                'kd_min': 0.001, 'kd_max': 0.003
            }
        elif target_rpm <= 4500:
            return {
                'kp_min': 0.007, 'kp_max': 0.009,
                'ki_min': 0.007, 'ki_max': 0.009,
                'kd_min': 0.003, 'kd_max': 0.007
            }
        else:
            return {
                'kp_min': 0.008, 'kp_max': 0.012,
                'ki_min': 0.008, 'ki_max': 0.01,
                'kd_min': 0.002, 'kd_max': 0.007
            }

    def run_pid_experiment(self, setpoint, kp, ki, kd, duration=8):
        print(f"   👉 Tes #{self.experiment_count + 1} | SP:{setpoint:.0f} | PID: {kp:.4f}, {ki:.4f}, {kd:.4f}")
        
        integral = 0
        prev_rpm = 0
        self.rpm_filtered = 0
        errors = []
        
        start_time = time.time()
        last_time = start_time
        
        # Logika Floor
        KICK_POWER = 83.0   
        SAFE_FLOOR = 35.0   
        RPM_ALIVE  = 500    
        
        try:
            while (time.time() - start_time) < duration:
                current_time = time.time()
                dt = current_time - last_time
                
                if dt >= 0.1: 
                    try: raw_rpm = self.lab.RPM
                    except: raw_rpm = 0
                    
                    self.rpm_filtered = (0.7 * self.rpm_filtered) + (0.3 * raw_rpm)
                    pv = self.rpm_filtered
                    
                    error = setpoint - pv
                    P = kp * error
                    
                    potential_integral = integral + (error * dt)
                    d_rpm = (pv - prev_rpm) / dt if dt > 0 else 0
                    D = -kd * d_rpm 
                    
                    # DYNAMIC SAFE FLOOR
                    if setpoint < 2200:
                        current_floor = 65.0 
                    if setpoint < 3500:
                        current_floor = 60.0 
                    if setpoint < 4500:
                        current_floor = 57.0 
                    else:
                        current_floor = 55.0
                    
                    # Cek Output Sementara
                    op_temp = P + (ki * potential_integral) + D
                    
                    # Clamp Integral (Pakai current_floor)
                    if op_temp > 100.0 or (op_temp < current_floor and error < 0):
                        pass 
                    else:
                        integral = potential_integral
                        
                    I = ki * integral
                    op = P + I + D
                    
                    # LOGIKA HYBRID
                    if pv < RPM_ALIVE and op > 1.0:
                        op = max(op, KICK_POWER) 
                    else:
                        if op > 1.0:
                            op = max(op, current_floor)
                    
                    op = max(0.0, min(100.0, op))
                    self.lab.op(op)
                    
                    if (time.time() - start_time) > 1.5:
                        errors.append(error)
                        
                    prev_rpm = pv
                    last_time = current_time
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.lab.op(0); return None
            
        self.lab.op(0)
        
        if len(errors) > 5:
            mae = np.mean(np.abs(errors))
            zero_crossings = np.sum(np.diff(np.sign(errors)) != 0)
            
            metrics = {
                'setpoint': setpoint, 'kp': kp, 'ki': ki, 'kd': kd,
                'mae': mae, 'oscillations': zero_crossings, 'overshoot_pct': 0 
            }
            self.training_data.append(metrics)
            print(f"      ✅ MAE: {mae:.1f} | Osilasi: {zero_crossings}")
            self.experiment_count += 1

    def save_to_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pid_training_SMART_TIERED_{timestamp}.csv"
        keys = self.training_data[0].keys()
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.training_data)
        print(f"\n💾 DATA CERDAS TERSIMPAN: {filename}")
        return filename

if __name__ == "__main__":
    c = SmartTieredCollector()
    if c.connect():
        try:
            print("\n=== KOLEKSI DATA CERDAS BERTINGKAT (ADAPTIVE RANGES) ===")
            print("Setiap tingkatan RPM memiliki rentang parameter sendiri.")
            
            # DAFTAR TARGET (STAGES)
            stages = [2000, 3000, 4000, 5000]
            samples_per_stage = 50
            
            for stage in stages:
                print(f"\nMEMULAI TAHAP: {stage} RPM...")
                
                r = c.get_search_range(stage)
                print(f"   Range Kp: {r['kp_min']} - {r['kp_max']}")
                
                for i in range(samples_per_stage):
                    # Random di dalam range spesifik
                    kp = np.random.uniform(r['kp_min'], r['kp_max'])
                    ki = np.random.uniform(r['ki_min'], r['ki_max'])
                    kd = np.random.uniform(r['kd_min'], r['kd_max'])
                    
                    # Variasi target di sekitar stage
                    sp = stage + np.random.uniform(-150, 150)
                    
                    c.run_pid_experiment(sp, kp, ki, kd)
                    time.sleep(1.5) 
                    
        except KeyboardInterrupt: pass
        c.save_to_csv()
        c.lab.close()