import numpy as np
import fremen_whyte as fremen
import matplotlib.pyplot as plt
from time import time

"""
T = np.array([1585239556.828, 1585034743.188, 1585056902.313, 1585113061.208, 1585160359.236, 1585216693.622, 1585375374.679, 1585399110.379, 1585399721.471, 1585399973.943, 1585400142.006, 1585454365.69, 1585494298.554, 1585495193.539, 1585558294.455, 1585558509.831, 1585737622.403, 1585759277.667, 1585808558.864, 1585835899.392, 1585838314.179, 1585904063.589, 1585927471.154, 1585927632.077, 1586006920.714])
S = np.array([4, 4, 3, 2, 1, 5, 3, 4, 4, 5, 4, 2, 2, 3, 2, 4, 3, 2, 4, 3, 3, 3, 4, 3, 3])
"""

T = [1585239556.828, 1585034743.188, 1585056902.313, 1585113061.208, 1585160359.236, 1585216693.622, 1585375374.679, 1585399110.379, 1585399721.471, 1585399973.943, 1585400142.006, 1585454365.69, 1585494298.554, 1585495193.539, 1585558294.455, 1585558509.831, 1585737622.403, 1585759277.667, 1585808558.864, 1585835899.392, 1585838314.179, 1585904063.589, 1585927471.154, 1585927632.077, 1586006920.714] * 1000
S = [4, 4, 3, 2, 1, 5, 3, 4, 4, 5, 4, 2, 2, 3, 2, 4, 3, 2, 4, 3, 3, 3, 4, 3, 3] * 1000
T = np.array(T)
S = np.array(S)

frm = fremen.Fremen()
start = time()
frm = frm.fit(times=T, values=S, no_freqs=5, longest=86400, shortest=8640)
#frm = frm.fit(times=T, values=S, no_freqs=5, longest=604800, shortest=8640)
#frm = frm.fit(times=T, values=S, no_freqs=5, longest=2419200)
finish = time()
print('fit: ' + str(finish-start))


#testing = np.arange(604800)
testing = np.arange(86400) + 1586217600
#testing = np.array([604800])
#testing = np.array(604800)
#testing = 604800

start = time()
pred = frm.predict(testing)
finish = time()
print('predict: ' + str(finish-start))


plt.plot(np.array(pred))
plt.show()
plt.close()

print(np.pi*2.0/frm.omegas)