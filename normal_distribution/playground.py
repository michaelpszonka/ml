import matplotlib.pyplot as plt
import math

def normal_distribution(x, mu = 0, sigma = 1):
    return math.exp(-(x-mu)**2 / 2 / sigma**2) / math.sqrt(2*math.pi * sigma)


xs = [x /10.0 for x in range(-50, 50)]
plt.plot(xs, [normal_distribution(x, sigma = 1) for x in xs], '-', label=',mu=0, sigma=1')
plt.plot(xs, [normal_distribution(x, sigma = 2) for x in xs], ':', label=',mu=0, sigma=2')
plt.plot(xs, [normal_distribution(x, sigma = 0.5) for x in xs], '-.', label=',mu=0, sigma=0.5')

plt.legend()
plt.title('Various Normal Pdfs')
plt.show()