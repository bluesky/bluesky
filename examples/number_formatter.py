from random import random
from lmfit.lineshapes import gaussian
import numpy as np


def format_num(x, max_len=10, pre=5, post=5):

    if x > 10**pre or x < 10^-post:
        x = '%.{}e'.format(post) % x
    else:
        x = '%{}.{}f'.format(pre, post) % x

    return x

dataset = [gaussian(i) for i in np.linspace(-10, 10, 100)]
for d in dataset:
    print(format_num(d))
