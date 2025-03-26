import sys
import random
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                            QPushButton, QLineEdit, QHBoxLayout)
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

class GaugeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0  # Range 0-255

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        size = min(rect.width(), rect.height())
        painter.translate(rect.center())
        painter.scale(size / 200.0, size / 200.0)
        painter.translate(-100, -100)
        
        # Draw background arc
        painter.setBrush(Qt.GlobalColor.black)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawPie(20, 20, 160, 160, 45 * 16, 270 * 16)
        
        # Draw active arc (scale to 0-255)
        painter.setBrush(QColor(50, 200, 50))
        painter.setPen(QColor(50, 200, 50))
        angle = int((self.value / 255) * 270)
        painter.drawPie(20, 20, 160, 160, (45 + (270 - angle)) * 16, angle * 16)
        
        # Draw text
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(90, 110, f"{self.value}")

    def setValue(self, value):
        self.value = value
        self.update()

class RobotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Control GUI")
        self.setGeometry(100, 100, 800, 600)
        
        main_layout = QVBoxLayout()
        
        # Top Section: Gauges and Sensors
        top_layout = QHBoxLayout()
        
        # Left Gauge
        self.left_gauge = GaugeWidget()
        self.left_gauge.setFixedSize(150, 150)
        top_layout.addWidget(self.left_gauge)
        
        # Sensor Grid (Middle)
        sensor_layout = QVBoxLayout()
        sensor_label = QLabel("Sensor Readings")
        sensor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sensor_layout.addWidget(sensor_label)
        
        self.sensors = [QLabel("○") for _ in range(15)]
        sensor_row = QHBoxLayout()
        for sensor in self.sensors:
            sensor_row.addWidget(sensor)
        sensor_layout.addLayout(sensor_row)
        top_layout.addLayout(sensor_layout)
        
        # Right Gauge
        self.right_gauge = GaugeWidget()
        self.right_gauge.setFixedSize(150, 150)
        top_layout.addWidget(self.right_gauge)
        
        main_layout.addLayout(top_layout)
        
        # Middle Section: Command Input
        command_layout = QVBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter command for robot")
        self.send_button = QPushButton("Send Command")
        command_layout.addWidget(self.command_input)
        command_layout.addWidget(self.send_button)
        main_layout.addLayout(command_layout)
        
        # Bottom Section: Graphs
        graph_layout = QHBoxLayout()
        self.left_graph = pg.PlotWidget()
        self.right_graph = pg.PlotWidget()
        graph_layout.addWidget(self.left_graph)
        graph_layout.addWidget(self.right_graph)
        main_layout.addLayout(graph_layout)
        
        self.setLayout(main_layout)
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sensors)
        self.timer.start(1000)

    def update_sensors(self):
        # Update motor gauges (0-255)
        self.left_gauge.setValue(random.randint(0, 255))
        self.right_gauge.setValue(random.randint(0, 255))
        
        # Update sensor dots
        for sensor in self.sensors:
            if random.random() > 0.5:
                sensor.setText("●")
                sensor.setStyleSheet("color: blue;")
            else:
                sensor.setText("○")
                sensor.setStyleSheet("color: black;")
        
        # Update graphs
        self.left_graph.plot([random.randint(0, 100) for _ in range(10)], clear=True)
        self.right_graph.plot([random.randint(0, 100) for _ in range(10)], clear=True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RobotGUI()
    gui.show()
    sys.exit(app.exec())