# ===============================================================
#  PANEL BIOMÉDICO — ECG (gráfico) + SpO₂ + HR + CAÍDAS + EPILEPSIA
#  Interfaz moderna estilo Dashboard (PyQt5 + MatPlotLib)
# ===============================================================

import sys, time, threading, queue, csv
from collections import deque
from datetime import datetime

import numpy as np
import serial
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------

DEFAULT_PORT = "COM7"
BAUDRATE = 115200

ECG_BUFFER = 3000
PLOT_LEN = 800

ECG_HP_WINDOW = 160
ECG_SMOOTH_WINDOW = 5
ECG_PEAK_STD_FACTOR = 1.0
ECG_MIN_RR_MS = 350

SPO2_WINDOW = 200
SPO2_MIN_VARIATION = 50

ACC_SMOOTH_WINDOW = 4
FALL_ACCEL_THRESHOLD = 1200
FALL_IMMOBILITY_SEC = 2
SEIZURE_TREMOR_THRESHOLD = 250

CSV_FLUSH_INTERVAL = 2.0


# ------------------------------------------------------
# FUNCIONES ÚTILES
# ------------------------------------------------------
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def moving_average(arr, window):
    if window <= 1 or len(arr) < 2:
        return np.array(arr, dtype=float)
    w = min(window, len(arr))
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode='same')


def highpass_subtract_mavg(signal, window):
    if len(signal) < 3:
        return np.array(signal, dtype=float)
    window = max(1, min(window, len(signal)-1))
    arr = np.array(signal, dtype=float)
    kernel = np.ones(window) / window
    mavg = np.convolve(arr, kernel, mode='same')
    return arr - mavg


# ------------------------------------------------------
# SERIAL READER THREAD
# ------------------------------------------------------
class SerialReader(threading.Thread):
    def __init__(self, port, baud, q, stop_event):
        super().__init__(daemon=True)
        self.port, self.baud, self.q, self.stop_event = port, baud, q, stop_event
        self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(0.8)
            self.q.put(("__INFO__", f"Conectado {self.port}"))
        except Exception as e:
            self.q.put(("__ERROR__", str(e)))
            return

        while not self.stop_event.is_set():
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self.q.put(("__LINE__", line))
            except:
                break

        try: self.ser.close()
        except: pass

        self.q.put(("__INFO__", "Serial detenido"))


# ------------------------------------------------------
# DETECTOR HR (PICOS DINÁMICOS)
# ------------------------------------------------------
class HRDetector:
    def __init__(self):
        self.last_peak_t = None
        self.rr_intervals = deque(maxlen=8)

    def detect_peaks_dynamic(self, signal, timestamps_ms):
        peaks = []
        if len(signal) < 5:
            return []

        mean = np.mean(signal)
        std = np.std(signal)
        thresh = mean + ECG_PEAK_STD_FACTOR * std

        for i in range(1, len(signal)-1):
            v = signal[i]
            if v > thresh and v > signal[i-1] and v >= signal[i+1]:
                t = timestamps_ms[i]
                if self.last_peak_t is None or (t - self.last_peak_t) > ECG_MIN_RR_MS:
                    peaks.append((v, t))
                    self.last_peak_t = t

        return peaks

    def add_peak_time(self, t_ms):
        if self.last_peak_t is not None:
            rr = t_ms - self.last_peak_t
            if rr > 0:
                self.rr_intervals.append(rr)
        self.last_peak_t = t_ms

    def get_bpm(self):
        if len(self.rr_intervals) == 0:
            return None
        avg_rr = sum(self.rr_intervals) / len(self.rr_intervals)
        bpm = int(60000 / avg_rr)
        return bpm if 30 <= bpm <= 200 else None


# ------------------------------------------------------
# SPO2 CALCULATION (AC/DC)
# ------------------------------------------------------
def estimate_spo2_from_ir_red(ir_vals, red_vals):
    try:
        ir = np.array(ir_vals, float)
        red = np.array(red_vals, float)

        if len(ir) < 40:
            return None

        ac_ir = np.max(ir) - np.min(ir)
        ac_red = np.max(red) - np.min(red)

        dc_ir = np.mean(ir)
        dc_red = np.mean(red)

        if ac_ir < SPO2_MIN_VARIATION or ac_red < SPO2_MIN_VARIATION:
            return None

        r = (ac_red/dc_red) / ((ac_ir/dc_ir) + 1e-9)
        spo2 = 110 - 25*r
        return max(70, min(100, int(spo2)))
    except:
        return None


# =======================================================
#  DASHBOARD PRINCIPAL
# =======================================================
class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Panel Biomédico — ECG + SpO₂ + HR + Caídas + Epilepsia")
        self.resize(1250, 780)

        self.q = queue.Queue()
        self.stop_event = threading.Event()
        self.serial_thread = None

        self.ecg_buf = deque(maxlen=ECG_BUFFER)
        self.ecg_t = deque(maxlen=ECG_BUFFER)

        self.ir_buf = deque(maxlen=2000)
        self.red_buf = deque(maxlen=2000)

        self.ax_buf = deque(maxlen=500)
        self.ay_buf = deque(maxlen=500)
        self.az_buf = deque(maxlen=500)

        self.acc_window_ax = deque(maxlen=ACC_SMOOTH_WINDOW)
        self.acc_window_ay = deque(maxlen=ACC_SMOOTH_WINDOW)
        self.acc_window_az = deque(maxlen=ACC_SMOOTH_WINDOW)

        self.hr_detector = HRDetector()
        self.hr_from_ecg = None
        self.spo2_val = None

        self._fall_flag = False
        self._fall_time = None
        self._seizure_flag = False

        self.logging = False
        self.csv_file = None
        self.csv_writer = None
        self._last_csv_flush = time.time()

        # ACTUALIZACIONES VISIBLES CADA 5 s
        self._last_hr_update = time.time()
        self._last_spo2_update = time.time()

        self._build_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.poll_serial)
        self.timer.start(30)

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------
    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        main = QtWidgets.QVBoxLayout(w)

        # Top bar
        hb = QtWidgets.QHBoxLayout()
        main.addLayout(hb)

        hb.addWidget(QtWidgets.QLabel("Puerto:"))
        self.port_edit = QtWidgets.QLineEdit(DEFAULT_PORT)
        hb.addWidget(self.port_edit)

        self.btn_conn = QtWidgets.QPushButton("Conectar")
        self.btn_conn.clicked.connect(self.toggle_conn)
        hb.addWidget(self.btn_conn)

        self.btn_csv = QtWidgets.QPushButton("Iniciar CSV")
        self.btn_csv.setCheckable(True)
        self.btn_csv.clicked.connect(self.toggle_log)
        hb.addWidget(self.btn_csv)

        self.status_lbl = QtWidgets.QLabel("Estado: desconectado")
        hb.addWidget(self.status_lbl)
        hb.addStretch()

        # SUMMARY BAR
        summary = QtWidgets.QHBoxLayout()
        main.addLayout(summary)

        self.lbl_bpm = QtWidgets.QLabel("HR: — bpm")
        self.lbl_bpm.setStyleSheet("font-size:20pt; color:#00FFAA;")
        summary.addWidget(self.lbl_bpm)

        self.lbl_spo2 = QtWidgets.QLabel("SpO₂: — %")
        self.lbl_spo2.setStyleSheet("font-size:20pt; color:#00CCFF;")
        summary.addWidget(self.lbl_spo2)

        self.lbl_fall = QtWidgets.QLabel("Caída: —")
        self.lbl_fall.setStyleSheet("font-size:20pt; color:white;")
        summary.addWidget(self.lbl_fall)

        self.lbl_seizure = QtWidgets.QLabel("Convulsión: —")
        self.lbl_seizure.setStyleSheet("font-size:20pt; color:white;")
        summary.addWidget(self.lbl_seizure)

        summary.addStretch()

        # MAIN AREA
        area = QtWidgets.QHBoxLayout()
        main.addLayout(area)

        # ECG GRAPH
        fig = Figure(figsize=(9, 5), tight_layout=True)
        self.canvas = FigureCanvas(fig)
        self.ax_ecg = fig.add_subplot(111)
        area.addWidget(self.canvas, 1)

        # RIGHT PANEL
        rp = QtWidgets.QVBoxLayout()
        area.addLayout(rp)

        rp.addWidget(QtWidgets.QLabel("<b>Lecturas MAX30102</b>"))
        self.lbl_ir = QtWidgets.QLabel("IR: —")
        self.lbl_red = QtWidgets.QLabel("RED: —")
        self.lbl_ir_sensorval = QtWidgets.QLabel("SpO₂(sensor): —")
        self.lbl_hr_sensorval = QtWidgets.QLabel("HR(sensor): —")

        rp.addWidget(self.lbl_ir)
        rp.addWidget(self.lbl_red)
        rp.addWidget(self.lbl_ir_sensorval)
        rp.addWidget(self.lbl_hr_sensorval)

        rp.addSpacing(8)
        rp.addWidget(QtWidgets.QLabel("<b>Acelerómetro</b>"))
        self.lbl_ax = QtWidgets.QLabel("AX: —")
        self.lbl_ay = QtWidgets.QLabel("AY: —")
        self.lbl_az = QtWidgets.QLabel("AZ: —")

        rp.addWidget(self.lbl_ax)
        rp.addWidget(self.lbl_ay)
        rp.addWidget(self.lbl_az)
        rp.addStretch()

    # ------------------------------------------------------
    # Serial
    # ------------------------------------------------------
    def toggle_conn(self):
        if self.serial_thread and self.serial_thread.is_alive():
            self.stop_event.set()
            self.btn_conn.setText("Conectar")
            self.status_lbl.setText("Estado: desconectado")
        else:
            port = self.port_edit.text().strip()
            self.q = queue.Queue()
            self.stop_event.clear()
            self.serial_thread = SerialReader(port, BAUDRATE, self.q, self.stop_event)
            self.serial_thread.start()
            self.btn_conn.setText("Desconectar")
            self.status_lbl.setText(f"Conectando {port}...")

    def toggle_log(self):
        self.logging = self.btn_csv.isChecked()
        if self.logging:
            fname = f"biomed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.csv_file = open(fname, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["ts","ecg","bpm","spo2","ir","red","ax","ay","az"])
            self.btn_csv.setText("Detener CSV")
        else:
            if self.csv_file:
                self.csv_file.close()
            self.btn_csv.setText("Iniciar CSV")

    # ------------------------------------------------------
    # Procesar serial
    # ------------------------------------------------------
    def poll_serial(self):
        updated = False
        while not self.q.empty():
            typ, data = self.q.get()
            if typ == "__LINE__":
                self.process_line(data)
                updated = True
            elif typ == "__ERROR__":
                self.status_lbl.setText("ERROR: " + data)
                self.btn_conn.setText("Conectar")

        if updated:
            self.update_ecg_plot()

    def process_line(self, line):

        parts = [p.strip() for p in line.split("|")]
        ecg = ir = red = hr = spo2 = ax = ay = az = None

        for p in parts:
            if ":" not in p:
                continue
            k, v = p.split(":", 1)
            k = k.strip().upper()
            v = v.strip()
            try:
                if k == "ECG": ecg = float(v)
                elif k == "IR": ir = float(v)
                elif k == "RED": red = float(v)
                elif k == "HR": hr = float(v)
                elif k == "SPO2": spo2 = float(v)
                elif k == "AX": ax = float(v)
                elif k == "AY": ay = float(v)
                elif k == "AZ": az = float(v)
            except:
                pass

        t_ms = int(time.time()*1000)

        if ecg is not None:
            self.ecg_buf.append(ecg)
            self.ecg_t.append(t_ms)

        if ir is not None:
            self.ir_buf.append(ir)
            self.lbl_ir.setText(f"IR: {int(ir)}")

        if red is not None:
            self.red_buf.append(red)
            self.lbl_red.setText(f"RED: {int(red)}")

        if hr is not None:
            self.lbl_hr_sensorval.setText(f"HR(sensor): {int(hr)}")

        if spo2 is not None:
            self.lbl_ir_sensorval.setText(f"SpO₂(sensor): {int(spo2)}%")

        if ax is not None:
            self.ax_buf.append(ax)
            self.acc_window_ax.append(ax)
            ax_s = sum(self.acc_window_ax) / len(self.acc_window_ax)
            self.lbl_ax.setText(f"AX: {ax_s:.1f}")

        if ay is not None:
            self.ay_buf.append(ay)
            self.acc_window_ay.append(ay)
            ay_s = sum(self.acc_window_ay) / len(self.acc_window_ay)
            self.lbl_ay.setText(f"AY: {ay_s:.1f}")

        if az is not None:
            self.az_buf.append(az)
            self.acc_window_az.append(az)
            az_s = sum(self.acc_window_az) / len(self.acc_window_az)
            self.lbl_az.setText(f"AZ: {az_s:.1f}")

        self.compute_ecg_hr()
        self.compute_spo2()
        self.detect_fall()
        self.detect_seizure()

    # ------------------------------------------------------
    # CÁLCULO HR ECG (SUAVIZADO CADA 5 s)
    # ------------------------------------------------------
    def compute_ecg_hr(self):
        if len(self.ecg_buf) < 150:
            return

        seg = np.array(list(self.ecg_buf)[-PLOT_LEN:], float)

        hp = max(1, min(ECG_HP_WINDOW, len(seg)-1))
        filt = highpass_subtract_mavg(seg, hp)
        filt = moving_average(filt, ECG_SMOOTH_WINDOW)

        if len(self.ecg_t) >= len(filt):
            ts = np.array(list(self.ecg_t)[-len(filt):], int)
        else:
            now = int(time.time()*1000)
            ts = np.arange(now - len(filt)*50, now, 50)

        peaks = self.hr_detector.detect_peaks_dynamic(filt, ts)
        for val, t in peaks:
            self.hr_detector.add_peak_time(t)

        bpm = self.hr_detector.get_bpm()

        if bpm:
            now_s = time.time()
            if now_s - self._last_hr_update >= 5:
                self.hr_from_ecg = bpm
                self.lbl_bpm.setText(f"HR: {bpm} bpm")
                self._last_hr_update = now_s

        self._last_ecg_plot = filt

    # ------------------------------------------------------
    # SpO₂ (SUAVIZADO CADA 5 s)
    # ------------------------------------------------------
    def compute_spo2(self):
        if len(self.ir_buf) < SPO2_WINDOW or len(self.red_buf) < SPO2_WINDOW:
            return

        ir_w = list(self.ir_buf)[-SPO2_WINDOW:]
        red_w = list(self.red_buf)[-SPO2_WINDOW:]

        spo2 = estimate_spo2_from_ir_red(ir_w, red_w)

        if spo2:
            now_s = time.time()
            if now_s - self._last_spo2_update >= 5:
                self.spo2_val = spo2
                self.lbl_spo2.setText(f"SpO₂: {spo2} %")
                self._last_spo2_update = now_s

    # ------------------------------------------------------
    # DETECT FALL
    # ------------------------------------------------------
    def detect_fall(self):
        if len(self.ax_buf) < 3:
            return

        ax = self.ax_buf[-1]
        ay = self.ay_buf[-1]
        az = self.az_buf[-1]

        magnitude = abs(ax) + abs(ay) + abs(az)

        if magnitude > FALL_ACCEL_THRESHOLD:
            self._fall_flag = True
            self._fall_time = time.time()
            self.lbl_fall.setText("¡POSIBLE CAÍDA!")
            self.lbl_fall.setStyleSheet("color: orange; font-weight:bold;")
            return

        if self._fall_flag:
            if (time.time() - self._fall_time) > FALL_IMMOBILITY_SEC:
                var = (
                    np.var(list(self.ax_buf)[-40:]) +
                    np.var(list(self.ay_buf)[-40:]) +
                    np.var(list(self.az_buf)[-40:])
                )

                if var < 300:
                    self.lbl_fall.setText("CAÍDA CONFIRMADA")
                    self.lbl_fall.setStyleSheet("color: red; font-weight:bold;")
                else:
                    self.lbl_fall.setText("Caída no confirmada")
                    self.lbl_fall.setStyleSheet("color: gray;")

                self._fall_flag = False

    # ------------------------------------------------------
    # DETECT EPILEPSIA
    # ------------------------------------------------------
    def detect_seizure(self):
        if len(self.ax_buf) < 30:
            return

        mov = (
            np.std(list(self.ax_buf)[-25:]) +
            np.std(list(self.ay_buf)[-25:]) +
            np.std(list(self.az_buf)[-25:])
        )

        rms = np.sqrt(np.mean([
            self.ax_buf[-1]**2,
            self.ay_buf[-1]**2,
            self.az_buf[-1]**2
        ]))

        if mov > SEIZURE_TREMOR_THRESHOLD and rms > 350:
            self.lbl_seizure.setText("¡CONVULSIÓN!")
            self.lbl_seizure.setStyleSheet("color: red; font-weight:bold;")
        else:
            self.lbl_seizure.setText("Estable")
            self.lbl_seizure.setStyleSheet("color: lightgreen;")

    # ------------------------------------------------------
    # PLOT ECG
    # ------------------------------------------------------
    def update_ecg_plot(self):
        self.ax_ecg.cla()

        if hasattr(self, "_last_ecg_plot"):
            data = self._last_ecg_plot
        else:
            data = list(self.ecg_buf)[-PLOT_LEN:]

        if len(data) > 5:
            x = np.arange(len(data))
            self.ax_ecg.plot(x, data, linewidth=1.4, color="#00FFFF")

            ymin = np.min(data) - 30
            ymax = np.max(data) + 30
            self.ax_ecg.set_ylim(ymin, ymax)

        self.ax_ecg.set_facecolor("#000000")
        self.ax_ecg.tick_params(colors="#00CCFF")
        self.ax_ecg.grid(color="#003344", linestyle="dotted")

        self.canvas.draw()


# =======================================================
# MAIN
# =======================================================
def main():
    app = QtWidgets.QApplication(sys.argv)

    app.setStyleSheet("""
        QWidget {
            background-color: #0A0F12;
            color: #D0F0FF;
            font-size: 12pt;
            font-family: 'Segoe UI';
        }
        QLabel {
            color: #C0FFFF;
        }
        QPushButton {
            background-color: #003344;
            padding: 8px;
            border-radius: 6px;
            color: white;
        }
        QPushButton:hover {
            background-color: #005577;
        }
        QLineEdit {
            background-color: #001a26;
            padding: 6px;
            border: 1px solid #004455;
            border-radius: 4px;
            color: #BBFFFF;
        }
    """)

    win = DashboardWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
