"""
A test to make sure the classification of last position is correct.

Author: Hanna Norberg
e-mail: hanna.gjelstrup.norberg@gmail.com
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

# -----------------------------
# Geometry parameters
# -----------------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CX, CY = 320, 480       # Ellipse center
A = 125                 # Half width
B = 60                  # Half height


def point_in_half_ellipse(x, y):
    """
    Returns True if (x, y) is inside the upper half
    of the ellipse.
    """
    if y > CY:
        return False

    return ((x - CX) ** 2) / (A ** 2) + ((y - CY) ** 2) / (B ** 2) <= 1


# -----------------------------
# Plot setup
# -----------------------------
fig, ax = plt.subplots()

# Draw image frame
frame = Rectangle((0, 0), FRAME_WIDTH, FRAME_HEIGHT, fill=False)
ax.add_patch(frame)

# Draw ellipse (lower half is outside the frame)
ellipse = Ellipse((CX, CY), width=2 * A, height=2 * B, fill=False)
ax.add_patch(ellipse)

# Axis configuration (image coordinates)
ax.set_xlim(0, FRAME_WIDTH)
ax.set_ylim(FRAME_HEIGHT, 0)  # invert Y-axis
ax.set_aspect('equal')

# Text feedback
status_text = ax.text(10, 20, "", fontsize=12)

# Store last plotted point so it can be cleared
last_point = None


def on_click(event):
    global last_point

    if event.inaxes != ax:
        return

    x, y = event.xdata, event.ydata
    inside = point_in_half_ellipse(x, y)

    # Remove previous point
    if last_point is not None:
        last_point.remove()

    # Plot clicked point
    last_point = ax.plot(x, y, 'o')[0]

    # Update text
    status_text.set_text(f"{inside}")

    fig.canvas.draw_idle()


# Connect mouse click event
fig.canvas.mpl_connect('button_press_event', on_click)

plt.title("Click to test point inside half-ellipse")
plt.show()
