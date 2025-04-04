import matplotlib.pyplot as plt

sensor_points = []
HORIZONAT_SENSOR_DISTANCE = 9 # horizontal distance between each sensor in mm
VERTICAL_SENSOR_DISTANCE = 5 # vertical distance between each sensor in mm

for i in range(15):
    if i < 4:
        sensor_points.append((i * HORIZONAT_SENSOR_DISTANCE, i*VERTICAL_SENSOR_DISTANCE))
    elif i >= 4 and i < 10:
        sensor_points.append((i * HORIZONAT_SENSOR_DISTANCE, 4*VERTICAL_SENSOR_DISTANCE))
    else:
        sensor_points.append((i * HORIZONAT_SENSOR_DISTANCE, (14-i)*(VERTICAL_SENSOR_DISTANCE)))

x_coords, y_coords = zip(*sensor_points)

print(sensor_points)
plt.scatter(x_coords, y_coords, color='blue', label='Sensor Points')
plt.grid(True)
plt.show()