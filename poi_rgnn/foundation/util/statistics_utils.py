import statistics as st
import math
import numpy as np

from .t_distribution import T_Distribution
import scipy.stats as stt

def t_distribution_test(x, confidence=0.95):

    n = len(x)
    mean = round(st.mean(x), 2)
    liberty_graus = n
    s = st.stdev(x)
    alfa = 1 - confidence
    column = 1 - alfa / 2
    t_value = T_Distribution().find_t_distribution(column, liberty_graus)
    average_variation = round(t_value * (s / math.pow(n, 1/2)), 2)
    average_variation = str(average_variation)
    if len(average_variation) == 3:
        average_variation = average_variation + "0"

    mean = str(mean)
    if len(mean) == 3:
        mean = mean + "0"

    ic = stt.t.interval(alpha=0.95, df=len(x) - 1, loc=np.mean(x), scale=stt.sem(x))
    l = round(ic[0], 2)
    r = round(ic[1], 2)
    library_variation = str(round(r - np.mean(x), 2))
    print("Library: ", library_variation, " local: ", average_variation)

    return str(mean) + u"\u00B1" + average_variation