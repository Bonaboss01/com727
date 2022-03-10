import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# generate dataset, make_forge() is a data set.
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()

# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(X.shape))
plt.show()