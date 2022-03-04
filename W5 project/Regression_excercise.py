
#Part-1
import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]
plt.plot(months, profit, "o")

plt.title("My Company-ABC")
plt.xlabel("Months")
plt.ylabel("Profit (£)")

plt.show()

#####################################################################################
#Part-2
import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]
#slope:

plt.plot(months, profit, "o")

plt.title("My Company-ABC")
plt.xlabel("Months")
plt.ylabel("Profit (£)")
plt.show()


# #####################################################################################
# # Part-3
x = [1, 2, 3]
y = [5, 1, 3]

# m1=1, b1=0

# predict y_predicted1


# y = 0.5x + 1


# predict y_predicted2

total_loss1 = 0
total_loss2=0

#use for loop to find total_loss1 and total_loss2


print(total_loss1)
print(total_loss2)

#decide using if statement and print which line is better fit



######################################################################################
#Part 4

def find_gradient_b(x, y, m, b):
    diff = 0
    N = len(x)
    for i in range(len(x)):
        diff = diff + (y[i] - (m * x[i] + b))
    b_gradient = -2 / N * diff
    return b_gradient
# similarly define function: find_gradient_m()




# #####################################################################################
# Part-5

def find_gradient_b(x, y, b, m):
    diff = 0
    N = len(x)
    for i in range(len(x)):
        diff = diff + (y[i] - (m * x[i] + b))
    b_gradient = -2 / N * diff
    return b_gradient


# similarly define function: find_gradient_m()


# #Your step_gradient function here
def step_gradient(x, y, b_current, m_current):
    b_gradient = find_gradient_b(x, y, b_current, m_current)
    #call find_gradient_m and store the returned value in m_gradient


    b = b_current - (0.01 * b_gradient)
    # similarly fine the new m valuefrom m_gradient = m_current - (0.01 * m_gradient)
   #m= complete this line

    return [b, m]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]

# current intercept guess:
b = 0
# current slope guess:
m = 0
# call step_gradient() using months, profit, and the inital guessed values of b and m


print(b, m)

# #####################################################################################
# Part-6
import matplotlib.pyplot as plt

def find_gradient_b(x, y, b,m):
    diff = 0
    N = len(x)
    for i in range(len(x)):
        diff = diff + (y[i] - (m * x[i] + b))
    b_gradient = -2 / N * diff
    return b_gradient

# similarly, define function: find_gradient_m()


# #Your step_gradient function here
def step_gradient(x, y, b_current, m_current,learning_rate):
    b_gradient = find_gradient_b(x, y, b_current, m_current)
    #uncomment the following line after you implment find_gradient_m
    #m_gradient = find_gradient_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]

#implment gradient_descent() function here


months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]

# current intercept guess:
b = 0
# current slope guess:
m = 0

learning_rate=0.005
iteration_num=1000
#uncomment the following line after you implement gradient_descent() function
#b, m = gradient_descent(months, profit, learning_rate, iteration_num)
print(b,m)

profit_predicted=[element*m+b for element in months]

plt.plot(months, profit, "o")
# add the profit_predicted to the same plot
# add legend, title, lables appropriately

###############################################################################
# Part-7
# import matplotlib.pyplot as plt
# import pandas as pd
# def find_gradient_b(x, y, b,m):
#     diff = 0
#     N = len(x)
#     for i in range(len(x)):
#         diff = diff + (y[i] - (m * x[i] + b))
#     b_gradient = -2 / N * diff
#     return b_gradient
#
# similarly, define function: find_gradient_m()


# #Your step_gradient function here
def step_gradient(x, y, b_current, m_current,learning_rate):
    b_gradient = find_gradient_b(x, y, b_current, m_current)
    #uncomment the following line after you implment find_gradient_m
    #m_gradient = find_gradient_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]

#implment gradient_descent() function here

# df = pd.read_csv("weight-height.csv")
# X_data = df['Height']
# Y_data = df['Weight']
#
# # current intercept guess:
# b = 0
# # current slope guess:
# m = 0
#
# learning_rate=0.0001
# iteration_num=1000
#
# b, m = gradient_descent(X_data, Y_data, learning_rate, iteration_num)
# print(b,m)
# weight_predicted=[element*m+b for element in X_data]
#
# plt.plot(X_data, Y_data, ".")
# plt.plot(X_data, weight_predicted,".")
# plt.title("My Company-ABC")
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.legend(['Observed Data', 'predicted Data'])
# plt.show()
#

##################################################################################
#part 8

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df = pd.read_csv("weight-height_full.csv")
X_data = df['Height'].to_numpy()
Y_data = df['Weight'].to_numpy()

X_data=X_data.reshape(-1,1)
plt.plot(X_data, Y_data, 'o')

# use LinearRegression object to fit model as explained on the slide

