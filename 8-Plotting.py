""" Description: Learning how to plot graphs in python. """

import matplotlib.pyplot as plt
import numpy as np 

class plot:
    def simple_plot(): 
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

        plt.plot(x, y)
        plt.show()

    def plot_without_lines():
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

        plt.plot(x, y, 'o')
        plt.show()
    
    def plot_with_labels(): 
        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])

        plt.plot(x, y)
       
        plt.title("My beautiful plot of Volatage - Current")
        plt.xlabel("My beautiful x axis in Volts")
        plt.ylabel("My beautiful y axies in Amperes")

        plt.show()
    
    def plotting_basic_equation():
        x = np.array(range(0, 2))

        a = 1 + (3 * x)
        b = 6 - (4 * x)
        c = 3 - (2 * x)
        d = 5 - (5 * x)

        plt.plot(x, a)
        plt.plot(x, b)
        plt.plot(x, c)
        plt.plot(x, d)

        plt.show()

# Execute 

#plot.simple_plot()
#plot.plot_without_lines()
#plot.plot_with_labels()
plot.plotting_basic_equation()
