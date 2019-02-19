import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.reset_orig()


def plot_a_vector(x, y, tick=2):
    """
    (x, y) 的向量
    """
    # 向量的起点
    x0 = 0
    y0 = 0
    # 指定向量的终点
    dx = x
    dy = y
    # 指定坐标轴范围
    max_val = max(abs(dx), abs(dy)) + 2
    plt.xlim(-max_val, max_val)
    plt.ylim(-max_val, max_val)
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    if dx >= 0:
        adjust = 0.5
    else:
        adjust = -3
    ax.annotate('({}, {})'.format(dx, dy), xy=(dx, dy), xytext=(dx + adjust, dy))
    base_color = sns.color_palette()[0]
    plt.arrow(x0, y0, dx, dy, length_includes_head=True, head_width=0.3, color=base_color)
    # 刻度
    plt.xticks(range(-max_val + 1, max_val - 1, tick))
    plt.yticks(range(-max_val + 1, max_val - 1, tick))
    plt.show()


def line_cross():
    x = np.linspace(0, 10, 10)
    y = (2 * x) + 1
    z = 0.1 * x + 6

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.plot(x, y, color="red", label='y=2x+1')
    plt.plot(x, z, "blue", label='y=1/10x+6')

    plt.legend()


def line_cross_point(m1x, n1y, k1, m2x, n2y, k2):
    # mx + by + k = 0
    m = m1x * n2y - n1y * m2x
    if m == 0:
        print("无交点")
    else:
        x = (k2 * n1y - k1 * n2y) / m
        y = (k1 * m2x - k2 * m1x) / m
    return [x, y]


def check_mat():
    a_matrix = np.array([[21, 7], [15, 42], [9, 6]])
    b_matrix = np.array([[4], [33]])
    origin_result = np.dot(a_matrix, b_matrix)
    mat_result = np.dot(np.mat(a_matrix), np.mat(b_matrix))
    print(mat_result == origin_result)


def check_add_dot():
    a_matrix_2 = np.array([[9, 3], [8, 4], [7, 6]])
    b_matrix_2 = np.array([[5], [2]])
    c_matrix_2 = np.array([[5], [5]])
    result_1 = np.dot(a_matrix_2, b_matrix_2 + c_matrix_2)
    result_2 = np.dot(a_matrix_2, b_matrix_2) + np.dot(a_matrix_2, c_matrix_2)
    print(result_1 == result_2)


def calc_Norm(x, p=2, infty=False):
    if infty:
        return float(np.max(np.abs(x)))
    if p == 0:
        return float(np.sum(np.array(x) > 0))
    elif p > 0:
        return float(np.power(np.sum(np.power(x, p)), 1 / p))
    else:
        raise Exception("范数的阶 >= 0")


def calc_Frobenius_Norm(array):
    np_array = None
    if isinstance(array, list):
        np_array = np.array(array)
    elif isinstance(array, np.ndarray):
        np_array = array
    if np_array is None:
        raise Exception("只支持list和numpy数组")
    return np.sqrt(np.sum(np.power(np_array, 2)))


def calc_svd(array):
    np_array = None
    if isinstance(array, list):
        np_array = np.array(array)
    elif isinstance(array, np.ndarray):
        np_array = array
    if np_array is None:
        raise Exception("只支持list和numpy数组")
    return np.linalg.svd(np_array)


if __name__ == '__main__':
    a_matrix_3 = np.array([[7, 6], [29, 3]])
    b_matrix_3 = np.array([[2, -8], [9, 10]])
    c_matrix_3 = np.array([[2, 17], [1, 5]])
    print(np.sum(np.dot(np.linalg.inv(a_matrix_3), a_matrix_3)))
