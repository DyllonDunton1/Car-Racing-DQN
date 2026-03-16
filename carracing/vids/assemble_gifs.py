import imageio.v2 as imageio
import numpy as np

gif1 = imageio.mimread("base_policy.gif", memtest=False)
gif2 = imageio.mimread("flipped.gif", memtest=False)

while len(gif2) < len(gif1):
    gif2.append(gif2[-1])

combined_frames = []
print(len(gif1), len(gif2))
for f1, f2 in zip(gif1, gif2):
    combined = np.hstack((f1, f2))   # side-by-side
    combined_frames.append(combined)

print(len(combined_frames))

imageio.mimsave("flip_comparison.gif", combined_frames, fps=45)