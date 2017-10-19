import matplotlib.pyplot as plt
import numpy as np
from commonLib import *

# plt.xticks([1,2,3,4,5],['$really\ bad$', '$bad$', '$normal$', '$good$', '$really\ good$'])
# plt.plot([3,4,5,6,7])
# plt.show()
countdic = {2:4,1:3,3:5}
dict = sorted(countdic.items(),key=lambda d:d[1])
write_dic(dict,'test')
a = read_dic('test')
print(dict==a)
print(a)
c = {7:4,9:1}
dict = sorted(c.items(),key=lambda d:d[1])
write_dic(dict,'test')
a = read_dic('test')
print(a)
