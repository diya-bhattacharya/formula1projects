import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import fastf1 as ff1

#Set parameters
year = 2025
wknd = 1
ses = 'Q'
driver = 'VER'
colormap = plt.cm.plasma

#Load session data
session = ff1.get_session(year, wknd, ses)
session.load()
weekend = session.event

#Get fastest lap data for the driver
laps = session.laps.pick_driver(driver).pick_fastest()

# Get telemetry data
x = laps.telemetry['X']
y = laps.telemetry['Y']
speed = laps.telemetry['Speed']

#Colour laps by speed
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

#Create plot
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle(f"{weekend.name} {year} - {driver} - Speed", size=24, y = 0.95)

#Plot background track line
ax.plot(x, y, color='lightgray', linewidth=1, zorder=1)

norm = plt.Normalize(speed.min(), speed.max())
lc = LineCollection(segments, cmap=colormap, norm=norm, linestyle='-', linewidth=5)
lc.set_array(speed)
ax.add_collection(lc)

# Colour bar
cbaxes = fig.add_axes([0.25, 0.05, 0.5, 0.051])
normlegend = mpl.colors.Normalize(vmin=speed.min(), vmax=speed.max())
mpl.colorbar.ColorbarBase(cbaxes, norm=normlegend, cmap=colormap, orientation="horizontal", label="Speed (km/h)")

# Adjust plot 
ax.axis("off")
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.12)
plt.show()