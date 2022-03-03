import numpy as np
import math
import matplotlib.pyplot as plt


def dot(a, b):
    a = np.mat(a)
    b = np.mat(b)
    z = a * b.T
    return z


def cos_similaity(a, b):
    z = dot(a, b)
    similaity = z / math.sqrt(dot(a, a) * dot(b, b))
    return similaity


def sim_matrix(feature):
    # input is list
    length = len(feature)
    matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            matrix[i][j] = cos_similaity(feature[feature[i], feature[j]])
    return matrix


def save_similaity_Matrix(feature, savepath, classes, title):
    conf_matrix = sim_matrix(feature)
    classNumber = len(feature)
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=-45)
    plt.yticks(tick_marks, classes)
    iters = np.reshape([[[i, j] for j in range(classNumber)] for i in range(classNumber)], (conf_matrix.size, 2))
    for i, j in iters:
        plt.text(j, i, format(round(conf_matrix[i, j], 2)), va='center', ha='center')
    plt.tight_layout()
    # plt.show()
    plt.savefig(savepath)
    plt.close()


if __name__ == "__main__":
    pass
