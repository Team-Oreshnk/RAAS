"""Visualization helpers for the integrated stack."""
import numpy as np
import matplotlib.pyplot as plt

def plot_occupancy(grid,width=20,length=40,resolution=.5,title="Occupancy"):
    plt.figure(figsize=(7,10))
    plt.imshow(grid,origin="lower",extent=[-width/2,width/2,0,length],
               vmin=0,vmax=2,interpolation="nearest")
    plt.xlabel("x = lateral (m)"); plt.ylabel("y = forward (m)"); plt.title(title)
    return plt.gca()

def plot_planner_result(occupancy,trajectory,origin_x,resolution,title="Planner"):
    plt.figure(figsize=(7,10))
    plt.imshow(occupancy,origin="lower",
               extent=[origin_x,origin_x+occupancy.shape[1]*resolution,
                       0,occupancy.shape[0]*resolution],
               vmin=0,vmax=2,interpolation="nearest")
    if trajectory is not None:
        plt.plot(trajectory[:,0],trajectory[:,1],linewidth=2)
    plt.xlabel("x = forward (m)"); plt.ylabel("y = left (m)"); plt.title(title)
    return plt.gca()
