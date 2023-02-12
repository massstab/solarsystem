#!/usr/bin/env python

import numpy as np
import pandas as pd

# File to read in 
filename = 'SolSystData.dat'


# Loading text files with NumPy
data1 = np.loadtxt(filename, delimiter=',', unpack=False,
        dtype={'names': ('planet', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'),
        'formats': ('S10', np.float, np.float, np.float, np.float, np.float, np.float, np.float)})

print(data1)

#sun = data1[0]
#mercury = data1[1]


# Loading text files with Pandas
df = pd.read_csv(filename, sep=',', header=None,
        names=['planet', 'mass', 'x', 'y', 'z', 'vx', 'vy', 'vz'])

# Data is now in a Pandas dataframe
print(df)
