import matplotlib.pyplot as plt
import json
f = open('time_series_data.json','r')
series_list = json.load(f)
offset = -0.5
for trial,series in enumerate(series_list):
    deadlines , rewards = zip( *series )
    color = 'r' if trial < 100 else 'b'
    f = lambda deadline : deadline + offset
    plt.plot( map( f , deadlines) , rewards , color)
    offset += 0.005
plt.xlabel('deadline')
plt.ylabel('reward')
plt.axis([0,50,-2,3])
plt.gca().invert_xaxis()
plt.grid()
plt.show()
print len(series_list)
