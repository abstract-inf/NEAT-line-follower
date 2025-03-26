import sys
import random
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QHBoxLayout, QGridLayout, QFrame
from PyQt6.QtGui import QPainter, QConicalGradient, QColor
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

class GaugeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 25  # Default value

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
        
        # Draw active arc
        painter.setBrush(QColor(50, 200, 50))
        painter.setPen(QColor(50, 200, 50))
        angle = int((self.value / 100) * 270)
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
        self.setGeometry(100, 100, 800, 500)
        
        main_layout = QVBoxLayout()
        
        # Sensor Readings
        sensor_label = QLabel("Sensor Readings")
        sensor_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(sensor_label)
        
        sensor_layout = QHBoxLayout()
        self.sensors = [QLabel("○") for _ in range(15)]
        for sensor in self.sensors:
            sensor_layout.addWidget(sensor)
        main_layout.addLayout(sensor_layout)
        
        # Motor Speeds
        motors_layout = QHBoxLayout()
        
        left_layout = QVBoxLayout()
        self.left_gauge = GaugeWidget()
        left_layout.addWidget(QLabel("Left Motor Speed"))
        left_layout.addWidget(self.left_gauge)
        left_layout.addWidget(QLabel("Network output left motor speed: 80"))
        motors_layout.addLayout(left_layout)
        
        right_layout = QVBoxLayout()
        self.right_gauge = GaugeWidget()
        right_layout.addWidget(QLabel("Right Motor Speed"))
        right_layout.addWidget(self.right_gauge)
        right_layout.addWidget(QLabel("Network output right motor speed: 80"))
        motors_layout.addLayout(right_layout)
        
        main_layout.addLayout(motors_layout)
        
        # Speed Over Time Graphs
        graph_layout = QHBoxLayout()
        self.left_graph = pg.PlotWidget()
        self.right_graph = pg.PlotWidget()
        graph_layout.addWidget(self.left_graph)
        graph_layout.addWidget(self.right_graph)
        main_layout.addLayout(graph_layout)
        
        # Command Input
        command_layout = QVBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Write a command to send to the robot")
        self.send_button = QPushButton("Send")
        command_layout.addWidget(self.command_input)
        command_layout.addWidget(self.send_button)
        
        main_layout.addLayout(command_layout)
        
        self.setLayout(main_layout)
        
        # Timer to update sensor values
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_sensors)
        self.timer.start(1000)
    
    def update_sensors(self):
        # Update motor speed gauges
        self.left_gauge.setValue(random.randint(0, 100))
        self.right_gauge.setValue(random.randint(0, 100))
        
        # Update sensor readings
        for i, sensor in enumerate(self.sensors):
            if random.random() > 0.5:
                sensor.setText("●")
                sensor.setStyleSheet("color: blue;")
            else:
                sensor.setText("○")
                sensor.setStyleSheet("color: black;")
        
        # Update speed graphs
        self.left_graph.plot([random.randint(0, 100) for _ in range(10)], clear=True)
        self.right_graph.plot([random.randint(0, 100) for _ in range(10)], clear=True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RobotGUI()
    gui.show()
    sys.exit(app.exec())