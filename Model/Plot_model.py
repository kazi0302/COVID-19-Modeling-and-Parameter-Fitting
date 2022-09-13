'''
Plot SEIAR model in the order of 1) Region 2) Population
'''

import matplotlib.pyplot as plt

def plot(combined, t):

    if len(combined) == 5:
        fig, ax = plt.subplots(figsize = (10,5))
        ax.set_title(f'SEIAR Model - ODE Simulation')
        ax.plot(t, combined[0], 'b-', alpha=0.5, lw=2, label='Susceptible')
        ax.plot(t, combined[1], 'g-', alpha=0.5, lw=2, label='Exposed')
        ax.plot(t, combined[2], 'r-', alpha=0.5, lw=2, label='Infected - Symptomatic')
        ax.plot(t, combined[3], 'm-', alpha=0.5, lw=2, label='Infected - Asymptomatic')
        ax.plot(t, combined[4], 'y-', alpha=0.5, lw=2, label='Recovered')
        ax.legend()
        ax.grid()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize = (10,5))
        ax.set_title(f'SEIAR Model - ODE Simulation')
        ax.plot(t, combined[0], 'b-', alpha=0.5, lw=2, label='Central')
        ax.plot(t, combined[1], 'g-', alpha=0.5, lw=2, label='East')
        ax.plot(t, combined[2], 'r-', alpha=0.5, lw=2, label='North Central')
        ax.plot(t, combined[3], 'm-', alpha=0.5, lw=2, label='North Coastal')
        ax.plot(t, combined[4], 'y-', alpha=0.5, lw=2, label='North Inland')
        ax.plot(t, combined[5], 'c-', alpha=0.5, lw=2, label='South')
        ax.legend()
        ax.grid()
        plt.show()