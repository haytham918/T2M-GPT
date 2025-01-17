# i3CE Conference Paper notes



# Ergo loss notes
- Got reba in torch format, Done
  - REBA score --> ergo risk level --> ergo loss (quadratic seems best, <2 low, >3 high)
  - Format as loss func
  - neck twisting? what is the threshold
- Get angles from SMPL vert or 22-joint 3D pose (for humanML3D)
  - The first 21 joints in SMPL and SMPL-H are identical. So just take the first 21x3 numbers in theta. Ignore the hands by ignoring the joints 22 and above.
  - Got a few figures online with joint index and location, checked with the HumanML3D dataset, all good
  - Turns out everything need projection, need to convert ergo3d into torch?
  - Or just simplify to 3D angles for now, many of the negative angle threshold are shared, e.g., +20 trunk == -20 trunk
- Action level
  - Ask Gunwoo to take a look at the action level, see if it is correct
  - Maybe I should use the sigmoid instead