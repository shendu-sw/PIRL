import numpy as np
import matplotlib.pyplot as plt


def layout_generate(nx=200, length=10, positions=[], default_powers=[], units=[], angles=[]):
    """
    generate default layout matrix of heat sources with default powers
    """
    assert (len(positions)==len(default_powers) == len(units))

    # TODO 给出每个组件具体位置时，生成对应 layout 图像
    # location = np.array(pos, dtype=np.float32)
    # intensity = np.array(powers, dtype=np.float32)
    if np.array(positions)[1,1] > length:
        positions = np.array(positions)
    else:
        positions = np.array(positions) / length * nx
    location = np.array([k for k in positions]) / nx * length
    intensity = np.array([k for k in default_powers])

    sizes = np.array(units)
    angle = np.array(angles)
    realsize = (
            np.reshape(np.cos(angle), [-1, 1]) * sizes
            + np.reshape(np.sin(angle), [-1, 1]) * sizes[:, ::-1]
        )
    #print(realsize)
    ele_length = length / nx
    ele_x = np.arange(ele_length / 2, length, ele_length)
    ele_y = ele_x
    f_layout = np.zeros((nx, nx))
    number = len(positions)
    #print(number)

    for i in range(nx):
        for j in range(nx):
            point = np.array([ele_x[i], ele_y[j]]).reshape(1, 2)
            u1 = np.repeat(point, number, axis=0)
            a1 = np.zeros((number, 1))
            b1 = np.zeros((number, 1))
            u2 = location
            a2 = realsize[:, 0].reshape(-1, 1) / 2
            b2 = realsize[:, 1].reshape(-1, 1) / 2
            #print(u1,a1,b1,u2,a2,b2)
            overlap = overlap_rec_rec(u1, a1, b1, u2, a2, b2)
            # print(overlap)
            if np.max(overlap) > 0:
                ind = np.argsort(-overlap.reshape(1, -1))  # 按照逆序排列并对应序号
                f_layout[i, j] = intensity[ind[0, 0]]
    return f_layout


def overlap_rec_rec(u1, a1, b1, u2, a2, b2):
    """
    可同时处理多组组件之间的干涉计算。
    :param : u1, u2 两组件中心点坐标 n*2
             a1, b1 组件1 长、宽的一半 n*1
             a2, b2 组件2 长、宽的一半 n*1
    :return : overlap_area 干涉面积 n*1
    """
    Phi1 = np.minimum(
        np.abs(u1[:, 0].reshape([-1, 1]) - u2[:, 0].reshape([-1, 1]))
        - a1.reshape([-1, 1])
        - a2.reshape([-1, 1]),
        0,
    )
    Phi2 = np.minimum(
        np.abs(u1[:, 1].reshape([-1, 1]) - u2[:, 1].reshape([-1, 1]))
        - b1.reshape([-1, 1])
        - b2.reshape([-1, 1]),
        0,
    )
    overlap_area = (-Phi1) * (-Phi2)
    return overlap_area


if __name__ == '__main__':

    #positions = [[100, 100], [50, 50], [25,25]]
    #default_powers = [[2000], [1000], [3000]]
    #units = [[2, 2], [1,1], [2,2]]
    #angles = [[0], [0], [0]]
    #length = 10
    #nx = 200

    nx = 200
    length = 0.1
    angles = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    units = [[0.012,0.012],[0.01,0.01],[0.015,0.015],[0.013,0.013],[0.013,0.013],[0.02,0.01],[0.016,0.016],[0.01,0.02],[0.02,0.02],[0.011,0.011]]  # 24
    default_powers=[[30000],[30000],[30000],[30000],[30000],[30000],[30000],[30000],[30000],[30000]]
    positions = [[ 0.01927952,0.09154201],[0.09288323, 0.07906539],[0.04514527,0.01451643],[0.08 ,0.025],[0.07232013 ,0.08852603],[0.03580174,0.033368],[0.02121442,0.06563024],[0.04655992,0.0794678],[0.06, 0.055],[0.02189589,0.01398727]]
    #positions = [[38,183],[185,158],[90,29],[160,50],[145,177],[72,67],[42,131],[93,159],[120,110],[44,28]]

    default_layout = layout_generate(positions=positions, default_powers=default_powers, units=units, angles=angles, length=length, nx=nx)
    # print(default_layout)

    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(default_layout)
    plt.colorbar(im)
    fig.savefig('default_layout.png', dpi=300)
