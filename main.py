from tkinter import *
from tkinter import messagebox

import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return 2 * y * x + 5 - x * x


def y_ivp(x, x0, y0):
    c1 = (y0 - (((x0 ** 2) - 4) / (2 * x0))) / (np.exp(x0 ** 2))
    return (((x ** 2) - 4) / (2 * x)) + c1 * np.exp(x ** 2)


def exact(y0, x0, xn, step):
    x_arr = np.arange(x0, xn + step, step)
    y = []
    for x in x_arr:
        y.append(y_ivp(x, x0, y0))
    return x_arr, y


def y_ivp_plot(y0, x0, xn, step):
    plt.figure(figsize=(8, 8))
    if x0 < 0 < xn:
        x_arr, y = exact(y0, x0, -step, step)

        plt.plot(x_arr, y, c='b', label='Exact')

        x_arr, y = exact(y0, step, xn, step)
        plt.plot(x_arr, y, c='b')
    elif x0 < 0 == xn:
        x_arr, y = exact(y0, x0, -step, step)
        plt.plot(x_arr, y, c='b', label='Exact')
    else:
        x_arr, y = exact(y0, x0, xn, step)
        plt.plot(x_arr, y, c='b', label='Exact')


def euler_method(y0, x0, xn, step):
    y = [y0]
    x_arr = np.arange(x0, xn + step, step)
    error = []
    for x in x_arr:
        y_n = y[-1] + step * f(x, y[-1])
        error.append(y_ivp(x, x0, y0) - y[-1])
        y.append(y_n)
    return x_arr, y, error


def euler_method_plot(y0, x0, xn, step):
    y_ivp_plot(y0, x0, xn, step)
    if x0 < 0 < xn:
        x_arr, y, error = euler_method(y0, x0, -step, step)
        plt.plot(x_arr, y[:len(x_arr)], c='m', label='Euler method')
        plt.plot(x_arr, error, 'r', label="Error")

        x_arr, y, error = euler_method(y0, step, xn, step)
        plt.plot(x_arr, y[:len(x_arr)], c='m')
        plt.plot(x_arr, error, 'r')
    elif x0 < 0 == xn:
        x_arr, y, error = euler_method(y0, x0, -step, step)
        plt.plot(x_arr, y[:len(x_arr)], c='m', label='Euler method')
        plt.plot(x_arr, error, 'r', label="Error")
    else:
        x_arr, y, error = euler_method(y0, x0, xn, step)
        plt.plot(x_arr, y[:len(x_arr)], c='m', label='Euler method')
        plt.plot(x_arr, error, 'r', label="Error")


def imp_euler(y0, x0, xn, step):
    y = [y0]
    x_arr = np.arange(x0, xn + step, step)
    error = []
    for x in x_arr:
        k1 = f(x, y[-1])
        k2 = f(x + step, y[-1] + step * k1)
        y_n = y[-1] + step * (k1 + k2) / 2
        error.append(y_ivp(x, x0, y0) - y[-1])
        y.append(y_n)
    return x_arr, y, error


def imp_euler_plot(y0, x0, xn, step):
    y_ivp_plot(y0, x0, xn, step)
    if x0 < 0 < xn:
        x_arr, y, error = imp_euler(y0, x0, -step, step)
        plt.plot(x_arr, y[:len(x_arr)], c='g', label='Improved Euler method')
        plt.plot(x_arr, error, 'r', label="Error")

        x_arr, y, error = imp_euler(y0, step, xn, step)
        plt.plot(x_arr, y[:len(x_arr)], c='g')
        plt.plot(x_arr, error, 'r')
    elif x0 < 0 == xn:
        x_arr, y, error = imp_euler(y0, x0, -step, step)
        plt.plot(x_arr, y[:len(x_arr)], c='g', label='Improved Euler method')
        plt.plot(x_arr, error, 'r', label="Error")
    else:
        x_arr, y, error = imp_euler(y0, x0, xn, step)
        plt.plot(x_arr, y[:len(x_arr)], c='g', label='Improved Euler method')
        plt.plot(x_arr, error, 'r', label="Error")


def runge_kutta(y0, x0, xn, step):
    y = [y0]
    x_arr = np.arange(x0, xn + step, step)
    error = []
    for x in x_arr:
        k1 = f(x, y[-1])
        k2 = f(x + step / 2, y[-1] + step * k1 / 2)
        k3 = f(x + step / 2, y[-1] + step * k2 / 2)
        k4 = f(x + step, y[-1] + step * k3)
        y_n = y[-1] + step * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        error.append(y_ivp(x, x0, y0) - y[-1])
        y.append(y_n)
    return x_arr, y, error


def runge_kutta_plot(y0, x0, xn, step):
    y_ivp_plot(y0, x0, xn, step)

    if x0 < 0 < xn:
        x_arr, y, error = runge_kutta(y0, x0, -step, step)
        plt.plot(x_arr, y[:len(x_arr)], c='y', label='Runge_Kutta method')
        plt.plot(x_arr, error, 'r', label="Error")

        x_arr, y, error = runge_kutta(y0, step, xn, step)
        plt.plot(x_arr, y[:len(x_arr)], c='y')
        plt.plot(x_arr, error, 'r')
    elif x0 < 0 == xn:
        x_arr, y, error = runge_kutta(y0, x0, -step, step)
        plt.plot(x_arr, y[:len(x_arr)], c='y', label='Runge_Kutta method')
        plt.plot(x_arr, error, 'r', label="Error")
    else:
        x_arr, y, error = runge_kutta(y0, x0, xn, step)
        plt.plot(x_arr, y[:len(x_arr)], c='y', label='Runge_Kutta method')
        plt.plot(x_arr, error, 'r', label="Error")


def all_methods_plot(y0, x0, xn, step):
    y_ivp_plot(y0, x0, xn, step)

    if x0 < 0 < xn:
        eu_x, eu_y, eu_error = euler_method(y0, x0, -step, step)
        imp_x, imp_y, imp_error = imp_euler(y0, x0, -step, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, x0, -step, step)

        plt.plot(eu_x, eu_y[:len(eu_x)], c='m', label='Euler')
        plt.plot(imp_x, imp_y[:len(imp_x)], c='g', label='Improved Euler')
        plt.plot(rk_x, rk_y[:len(rk_x)], c='y', label='Runge-Kutta')

        eu_x, eu_y, eu_error = euler_method(y0, step, xn, step)
        imp_x, imp_y, imp_error = imp_euler(y0, step, xn, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, step, xn, step)

        plt.plot(eu_x, eu_y[:len(eu_x)], c='m')
        plt.plot(imp_x, imp_y[:len(imp_x)], c='g')
        plt.plot(rk_x, rk_y[:len(rk_x)], c='y')
    elif x0 < 0 == xn:
        eu_x, eu_y, eu_error = euler_method(y0, x0, -step, step)
        imp_x, imp_y, imp_error = imp_euler(y0, x0, -step, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, x0, -step, step)

        plt.plot(eu_x, eu_y[:len(eu_x)], c='m', label='Euler')
        plt.plot(imp_x, imp_y[:len(imp_x)], c='g', label='Improved Euler')
        plt.plot(rk_x, rk_y[:len(rk_x)], c='y', label='Runge-Kutta')

    else:
        eu_x, eu_y, eu_error = euler_method(y0, x0, xn, step)
        imp_x, imp_y, imp_error = imp_euler(y0, x0, xn, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, x0, xn, step)

        plt.plot(eu_x, eu_y[:len(eu_x)], c='m', label='Euler')
        plt.plot(imp_x, imp_y[:len(imp_x)], c='g', label='Improved Euler')
        plt.plot(rk_x, rk_y[:len(rk_x)], c='y', label='Runge-Kutta')


def all_local_errors(y0, x0, xn, step):
    plt.figure(figsize=(8, 8))

    if x0 < 0 < xn:
        eu_x, eu_y, eu_error = euler_method(y0, x0, -step, step)
        imp_x, imp_y, imp_error = imp_euler(y0, x0, -step, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, x0, -step, step)

        plt.plot(eu_x, eu_error, '--', c='m', label='Euler method error')
        plt.plot(imp_x, imp_error, '--', c='g', label='Improoved Euler method error')
        plt.plot(rk_x, rk_error, '--', c='y', label='Runge-Kutta method error')

        eu_x, eu_y, eu_error = euler_method(y0, step, xn, step)
        imp_x, imp_y, imp_error = imp_euler(y0, step, xn, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, step, xn, step)

        plt.plot(eu_x, eu_error, '--', c='m')
        plt.plot(imp_x, imp_error, '--', c='g')
        plt.plot(rk_x, rk_error, '--', c='y')
    elif x0 < 0 == xn:
        eu_x, eu_y, eu_error = euler_method(y0, x0, -step, step)
        imp_x, imp_y, imp_error = imp_euler(y0, x0, -step, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, x0, -step, step)

        plt.plot(eu_x, eu_error, '--', c='m', label='Euler method error')
        plt.plot(imp_x, imp_error, '--', c='g', label='Improoved Euler method error')
        plt.plot(rk_x, rk_error, '--', c='y', label='Runge-Kutta method error')
    else:
        eu_x, eu_y, eu_error = euler_method(y0, x0, xn, step)
        imp_x, imp_y, imp_error = imp_euler(y0, x0, xn, step)
        rk_x, rk_y, rk_error = runge_kutta(y0, x0, xn, step)

        plt.plot(eu_x, eu_error, '--', c='m', label='Euler method error')
        plt.plot(imp_x, imp_error, '--', c='g', label='Improoved Euler method error')
        plt.plot(rk_x, rk_error, '--', c='y', label='Runge-Kutta method error')


def global_error(y0, x0, xn, step):
    plt.figure(figsize=(8, 8))

    n = np.arange(1, int((xn - x0) / step) + 1, 1)
    eu_gl_e = []
    imp_gl_e = []
    rk_gl_e = []
    for i in n:
        x, y = exact(y0, x0, xn, (xn - x0) / i)
        _, eu_y, _ = euler_method(y0, x0, xn, (xn - x0) / i)
        _, imp_y, _ = imp_euler(y0, x0, xn, (xn - x0) / i)
        _, rk_y, _ = runge_kutta(y0, x0, xn, (xn - x0) / i)
        eu_gl_e.append((np.array(y) - np.array(eu_y[:len(y)])).sum())
        imp_gl_e.append((np.array(y) - np.array(imp_y[:len(y)])).sum())
        rk_gl_e.append((np.array(y) - np.array(rk_y[:len(y)])).sum())

    plt.plot(n, eu_gl_e, '.-', c='m', label='Euler method')
    plt.plot(n, imp_gl_e, '.-', c='g', label='Improved Euler method ')
    plt.plot(n, rk_gl_e, '.-', c='y', label='Runge-Kutta method')


window = Tk()
window.title("Assignment")
window.geometry('350x355+200+300')

lbl1 = Label(window, text="x0 =")
lbl1.grid(column=1, row=0)

def_x0 = IntVar()
def_x0.set(0)
scale1 = Scale(window, from_=-10, to=10, length=200, variable=def_x0, orient=HORIZONTAL)
scale1.grid(column=2, row=0)

lbl2 = Label(window, text="y0 =")
lbl2.grid(column=1, row=1)

def_y0 = IntVar()
def_y0.set(1)
scale2 = Scale(window, from_=-10, to=10, length=200, variable=def_y0, orient=HORIZONTAL)
scale2.grid(column=2, row=1)

lbl3 = Label(window, text="X =")
lbl3.grid(column=1, row=2)

def_X = IntVar()
def_X.set(3)
scale3 = Scale(window, from_=-10, to=10, length=200, variable=def_X, orient=HORIZONTAL)
scale3.grid(column=2, row=2)

lbl4 = Label(window, text="step =")
lbl4.grid(column=1, row=3)

def_step = DoubleVar()
def_step.set(0.01)
scale4 = Scale(window, from_=0.01, to=1, resolution=0.01, length=200, variable=def_step, orient=HORIZONTAL)
scale4.grid(column=2, row=3)


def ivp_clicked():
    plt.close()
    if not check_for_error():
        y_ivp_plot(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.title("Exact solution")
        plt.show()


btn1 = Button(window, text="IVP", command=ivp_clicked)
btn1.grid(column=3, row=4, sticky=E)


def euler_clicked():
    plt.close()
    if not check_for_error():
        euler_method_plot(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.title('Euler method')
        plt.legend()
        plt.show()


btn2 = Button(window, text="Euler", command=euler_clicked)
btn2.grid(column=3, row=5, sticky=E)


def imp_euler_clicked():
    plt.close()
    if not check_for_error():
        imp_euler_plot(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.title('Improved Euler method')
        plt.legend()
        plt.show()


btn3 = Button(window, text="Imp. Euler", command=imp_euler_clicked)
btn3.grid(column=3, row=6, sticky=E)


def runge_kutta_clicked():
    plt.close()
    if not check_for_error():
        runge_kutta_plot(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.title('Runge_Kutta method')
        plt.legend()
        plt.show()


btn4 = Button(window, text="Runge Kutta", command=runge_kutta_clicked)
btn4.grid(column=3, row=7, sticky=E)


def all_clicked():
    plt.close()
    if not check_for_error():
        all_methods_plot(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.legend()
        plt.title('All methods graph')
        plt.show()


btn5 = Button(window, text="All plots", command=all_clicked)
btn5.grid(column=3, row=8, sticky=E)


def all_local_errors_clicked():
    plt.close()
    if not check_for_error():
        all_local_errors(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.legend()
        plt.title("Local errors")
        plt.show()


btn6 = Button(window, text="All local errors", command=all_local_errors_clicked)
btn6.grid(column=3, row=9, sticky=E)


def global_error_clicked():
    plt.close()
    if not check_for_error():
        global_error(int(scale2.get()), int(scale1.get()), int(scale3.get()), float(scale4.get()))
        plt.title("Graph of Global Truncation Errors")
        plt.legend()
        plt.show()


btn7 = Button(window, text="Global error", command=global_error_clicked)
btn7.grid(column=3, row=10, sticky=E)


def check_for_error():
    if scale1.get() == 0:
        messagebox.showerror("Error", "Division by zero")
        return True
    elif scale3.get() < scale1.get():
        messagebox.showerror("Error", "X must be greater than x0!")
        return True
    else:
        return False


def on_closing():
    plt.close()
    window.destroy()


window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()
