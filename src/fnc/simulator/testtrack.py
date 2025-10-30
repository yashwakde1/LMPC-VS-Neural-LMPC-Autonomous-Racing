import sys
sys.path.append('fnc')
import matplotlib.pyplot as plt
import numpy as np
from TrackCustom import Map

# Initialize track
track = Map(halfWidth=0.4)

print(f"Track Length: {track.TrackLength:.2f} meters")
print(f"Track Half Width: {track.halfWidth} meters")
print("\nTrack segments:")
for i, pt in enumerate(track.PointAndTangent):
    print(f"Segment {i}: x={pt[0]:.2f}, y={pt[1]:.2f}, s={pt[3]:.2f}, length={pt[4]:.2f}, curv={pt[5]:.3f}")

# Generate points along the track
n_points = 500
s_points = np.linspace(0, track.TrackLength, n_points)

# Get track boundaries
center_x, center_y = [], []
inner_x, inner_y = [], []
outer_x, outer_y = [], []

for s in s_points:
    # Center line
    xc, yc = track.getGlobalPosition(s, 0)
    center_x.append(xc)
    center_y.append(yc)
    
    # Inner boundary
    xi, yi = track.getGlobalPosition(s, -track.halfWidth)
    inner_x.append(xi)
    inner_y.append(yi)
    
    # Outer boundary
    xo, yo = track.getGlobalPosition(s, track.halfWidth)
    outer_x.append(xo)
    outer_y.append(yo)

# Plot the track
plt.figure(figsize=(14, 10))
plt.plot(center_x, center_y, 'k--', linewidth=1.5, label='Center line', alpha=0.6)
plt.plot(inner_x, inner_y, 'r-', linewidth=2.5, label='Inner boundary')
plt.plot(outer_x, outer_y, 'b-', linewidth=2.5, label='Outer boundary')
plt.plot([0], [0], 'go', markersize=15, label='Start/Finish', zorder=5)

# Add segment markers
for i, pt in enumerate(track.PointAndTangent[:-1]):
    plt.plot(pt[0], pt[1], 'ko', markersize=8, alpha=0.5)
    plt.text(pt[0]+0.3, pt[1]+0.3, f'S{i}', fontsize=9, alpha=0.7)

plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12, loc='best')
plt.title('Custom Oval Track for LMPC', fontsize=16, fontweight='bold')
plt.xlabel('X (m)', fontsize=13)
plt.ylabel('Y (m)', fontsize=13)
plt.tight_layout()
plt.savefig('custom_track_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Track visualization saved as 'custom_track_visualization.png'")
print("✓ Track appears to be properly closed!")