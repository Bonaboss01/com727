import matplotlib.pyplot as plt
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]

plt.plot(months, profit, "o")
plt.plot(13,200,"o")
plt.title("My Company-ABC")
plt.xlabel("Months")
plt.ylabel("Profit (£)")
plt.legend(['Observed Data', 'Estimated Values'])
plt.show()
#
#
# ####################################################################################
# Part-2
# import matplotlib.pyplot as plt
# months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]
# #slope:
# m = 11
# #intercept:
# b = 49
# y=[element*m+b for element in months]
# plt.plot(months, profit, "o")
# plt.plot(months,y,"-o")
# plt.title("My Company-ABC")
# plt.xlabel("Months")
# plt.ylabel("Profit (£)")
# plt.legend(['Observed Data', 'predicted Values'])
# plt.show()
#
#
# #####################################################################################
# # Part-3
# x = [1, 2, 3]
# y = [5, 1, 3]
#
# # y = x
# m1 = 1
# b1 = 0
# y_predicted1 = [element * m1 + b1 for element in x]
# # y = 0.5x + 1
# m2 = 0.5
# b2 = 1
# y_predicted2 = [element * m2 + b2 for element in x]
# total_loss1 = 0
# for i in range(len(y)):
#     total_loss1 = total_loss1 + (y[i] - y_predicted1[i]) ** 2
#
# total_loss2 = 0
# for i in range(len(y)):
#     total_loss2 = total_loss2 + (y[i] - y_predicted2[i]) ** 2
#
# print(total_loss1)
# better_fit = 0
# if total_loss1 > total_loss2:
#     better_fit = 2
# else:
#     better_fit = 1
#
# print(better_fit)
#
# #####################################################################################
# #Part 4
#
# def find_gradient_b(x, y, m, b):
#     diff = 0
#     N = len(x)
#     for i in range(len(x)):
#         diff = diff + (y[i] - (m * x[i] + b))
#     b_gradient = -2 / N * diff
#     return b_gradient
#
# def find_gradient_m(x, y, m, b):
#     diff = 0
#
#     N = len(x)
#     m_gradient = -2 / N
#     for i in range(N):
#         y_val = y[i]
#         x_val = x[i]
#         a = x_val * (y_val - (m * x_val + b))
#         diff = diff + a;
#     return diff * m_gradient
#
# #####################################################################################
# Part-5
#
# def find_gradient_b(x, y, b,m):
#     diff = 0
#     N = len(x)
#     for i in range(len(x)):
#         diff = diff + (y[i] - (m * x[i] + b))
#     b_gradient = -2 / N * diff
#     return b_gradient
#
# def find_gradient_m(x, y, b, m):
#     diff = 0
#
#     N = len(x)
#     m_gradient = -2 / N
#     for i in range(N):
#         y_val = y[i]
#         x_val = x[i]
#         a = x_val * (y_val - (m * x_val + b))
#         diff = diff + a;
#     return diff * m_gradient
#
#
# # #Your step_gradient function here
# def step_gradient(x, y, b_current, m_current):
#     b_gradient = find_gradient_b(x, y, b_current, m_current)
#     m_gradient = find_gradient_m(x, y, b_current, m_current)
#     b = b_current - (0.01 * b_gradient)
#     m = m_current - (0.01 * m_gradient)
#     return [b, m]
#
# months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]
#
# # current intercept guess:
# b = 0
# # current slope guess:
# m = 0
#
# b, m = step_gradient(months, profit, b, m)
# print(b, m)
#
# #####################################################################################
#Part-6
import matplotlib.pyplot as plt
import pandas as pd

def find_gradient_b(x, y, b,m):
    diff = 0
    N = len(x)
    for i in range(len(x)):
        diff = diff + (y[i] - (m * x[i] + b))
    b_gradient = -2 / N * diff
    return b_gradient

def find_gradient_m(x, y, b, m):
    diff = 0
    N = len(x)
    m_gradient = -2 / N
    for i in range(N):
        y_val = y[i]
        x_val = x[i]
        a = x_val * (y_val - (m * x_val + b))
        diff = diff + a;
    return diff * m_gradient


# #Your step_gradient function here
def step_gradient(x, y, b_current, m_current,learning_rate):
    b_gradient = find_gradient_b(x, y, b_current, m_current)
    m_gradient = find_gradient_m(x, y, b_current, m_current)
    b = b_current - (learning_rate * b_gradient)
    m = m_current - (learning_rate * m_gradient)
    return [b, m]

def gradient_descent(x,y,learning_rate,iterations_num):
    b=0
    m=0
    for it in range(iterations_num):
        [b,m]=step_gradient(x,y,b,m,learning_rate)
    return [b,m]

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
profit = [50, 73, 81, 93, 116, 111, 128, 127, 144, 149, 156, 184]

# current y- intercept guess:
b = 0
# current slope guess:
m = 0

learning_rate=0.005
iteration_num=1000

b, m = gradient_descent(months, profit, learning_rate, iteration_num)
print(b,m)
profit_predicted=[element*m+b for element in months]

plt.plot(months, profit, "o")
plt.plot(months, profit_predicted,"o")
plt.title("My Company-ABC")
plt.xlabel("Months")
plt.ylabel("Profit (£)")
plt.legend(['Observed Data', 'predicted Data'])
plt.show()

##############################################################################
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
# def find_gradient_m(x, y, b, m):
#     diff = 0
#     N = len(x)
#     m_gradient = -2 / N
#     for i in range(N):
#         y_val = y[i]
#         x_val = x[i]
#         a = x_val * (y_val - (m * x_val + b))
#         diff = diff + a;
#     return diff * m_gradient
#
#
# # #Your step_gradient function here
# def step_gradient(x, y, b_current, m_current,learning_rate):
#     b_gradient = find_gradient_b(x, y, b_current, m_current)
#     m_gradient = find_gradient_m(x, y, b_current, m_current)
#     b = b_current - (learning_rate * b_gradient)
#     m = m_current - (learning_rate * m_gradient)
#     return [b, m]
#
# def gradient_descent(x,y,learning_rate,iterations_num):
#     b=0
#     m=0
#     for it in range(iterations_num):
#         [b,m]=step_gradient(x,y,b,m,learning_rate)
#     return [b,m]
#
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
#
# #################################################################################
# part 8
#
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("weight-height_full.csv")
# X_data = df['Height'].to_numpy()
# Y_data = df['Weight'].to_numpy()
#
# X_data=X_data.reshape(-1,1)
# plt.plot(X_data, Y_data, 'o')
#
# line_fitter = LinearRegression()
# line_fitter.fit(X_data , Y_data)
# Y_data_predict=line_fitter.predict(X_data)
# plt.plot(X_data,Y_data_predict)
# plt.xlabel("Height")
# plt.ylabel("Weight")
# plt.legend(['Observed Data','Predicted Data']);
# plt.show()
#
