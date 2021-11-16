import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'DejaVu Sans',
        'size'   : 15}
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111) # 3 * 3 

x_array = [1,2,3]
y_array = [3,2,1]
ax.plot(x_array, y_array, color='r', label='some label')
ax.set_xlabel('x label')
ax.set_ylabel('y label')
ax.legend() 
# plt.show() 
plt.savefig('fig.pdf')
