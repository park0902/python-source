import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rawdate = pd.read_csv('d:\data7\\air.txt', names=['Mean', 'Max', 'Min', 'wind', 'maxwind', 'vis'])
print(rawdate.head(10))
print(rawdate.plot())