import matplotlib.pylab as plt
plt.ion()
plt.plot([1,2,3])
plt.show()
plt.ylabel('This is an axis')
print ("Hello")

#
#
#

from numpy import *
from matplotlib.pyplot import *
x = linspace(-2, 2, 100)
y2 = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*exp(-x**2)
y1 = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)*exp(-x)
y0 = (x**5 + 4*x**4 + 3*x**3 + 2*x**2 + x + 1)
figure()   # Make a new figure
plot(x, y2, label="$y_2(x)$")   # Plot some data
plot(x, y1, label="$y_1(x)$")   # Plot some data
plot(x, y0, label="$y_0(x)$")   # Plot some data
grid(True)   # Set the thin grid lines
ylim(0,6)   # Specify the range of the y axis shown
xlabel(r"$x$")   # Put a label on the x axis
ylabel(r"$y(x)$")   # Put a label on the y axis
legend(loc="lower right")   # Add a legend table
title("Compare "+r"$x^5+4x^4+3x^3+2x^2+x+1$"+" to $\exp(x^n)$")   # And a title on top of the figure

savefig("plot_1.png")


#
#
#

from scipy.special import gamma
from scipy.special.orthogonal import eval_hermite

x = linspace(-5, 5, 1000)

psi = lambda n,x: 1.0/sqrt((2**n*gamma(n+1)*sqrt(pi))) * exp(-x**2/2.0) * eval_hermite(n, x)

figure()
for n in xrange(4):
    plot(x, psi(n,x), label=r"$\psi_"+str(n)+r"(x)$")   # The 'label' to put into the legend

grid(True)

xlim(-5,5)   # Specify the range of the x axis shown
ylim(-0.8,0.8)   # Specify the range of the y axis shown
xlabel(r"$x$")   # Put a label on the x axis
ylabel(r"$\psi_n(x)$")   # Put a label on the y axis
legend(loc="lower right")   # Add a legend table
title(r"Hermite functions $\psi_n$")   # And a title on top of the figure
savefig("plot_2.png")