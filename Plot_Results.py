import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def Plot_Results():
    Algorithm = ['ÁLGORITHMS', 'JAYA', 'MFO', 'BWO', 'BA', 'PROPOSED']
    BestFit = np.load('Fitness.npy', allow_pickle=True)
    Time = np.load('Time.npy', allow_pickle=True)
    Size = np.load('Size.npy', allow_pickle=True)
    Makespan = np.load('Makespan.npy', allow_pickle=True)
    Energy = np.load('Energy.npy', allow_pickle=True)
    Res_Time = np.load('Res_Time.npy', allow_pickle=True)
    NoOfMachine = np.load('NoOfMachine.npy', allow_pickle=True)

    for i in range(BestFit.shape[0]):
        Fitness = BestFit[i, :, :]
        Fitness = np.squeeze(Fitness)

        Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('---------------------------------------- Configuration ' + str(i + 1) + ' Statistical Report ',
              '----------------------------------------')
        print(Table)

        for k in range(Fitness.shape[0]):
            Fitness[k, :] = np.sort(Fitness[k, :])[::-1]
        min_index = np.where(Fitness[:, Fitness.shape[1] - 1] == np.min(Fitness[:, Fitness.shape[1] - 1]))
        prop = min_index[0][0]
        normal = 4
        # x = Fitness[min_index[0][0], :]
        # y = Fitness[4, :]
        # x2 = y
        # Fitness[min_index[0][0], :] = x2
        # Fitness[4, :] = x

        length = np.arange(Fitness.shape[1])
        plt.plot(length, Fitness[0, :], color='r', linewidth=3, label="JA-AECC")
        plt.plot(length, Fitness[1, :], color='g', linewidth=3, label="MFO-AECC")
        plt.plot(length, Fitness[2, :], color='b', linewidth=3, label="BWO-AECC")
        plt.plot(length, Fitness[normal, :], color='m', linewidth=3, label="BA-AECC")
        plt.plot(length, Fitness[prop, :], color='k', linewidth=3, label="HBWBA-AECC")
        plt.ylabel('Cost Function')
        plt.xlabel('Iteration')
        # plt.xticks(length, (3, 6, 9))
        plt.legend(loc='best')
        plt.savefig('./Results/conv_' + str(i + 1) + '-' + '_line.png')
        plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(0, 10, 2)
    ax.bar(X + 0.00, NoOfMachine[:, 0], color='r', width=0.20, label="JA-AECC")
    ax.bar(X + 0.20, NoOfMachine[:, 1], color='g', width=0.20, label="MFO-AECC")
    ax.bar(X + 0.40, NoOfMachine[:, 2], color='b', width=0.20, label="BWO-AECC")
    ax.bar(X + 0.60, NoOfMachine[:, 3], color='m', width=0.20, label="BA-AECC")
    ax.bar(X + 0.80, NoOfMachine[:, 4], color='k', width=0.20, label="HBWBA-AECC")
    plt.ylabel('No of Active Machine')
    plt.xlabel('Configuration')
    plt.xticks(X + 0.20, (10, 20, 30, 40, 50))
    plt.legend(loc='best')
    path1 = './Results/Machine_bar.png'
    plt.savefig(path1)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(0, 10, 2)
    ax.bar(X + 0.00, Makespan[:, 0], color='r', width=0.20, label="JA-AECC")
    ax.bar(X + 0.20, Makespan[:, 1], color='g', width=0.20, label="MFO-AECC")
    ax.bar(X + 0.40, Makespan[:, 2], color='b', width=0.20, label="BWO-AECC")
    ax.bar(X + 0.60, Makespan[:, 3], color='m', width=0.20, label="BA-AECC")
    ax.bar(X + 0.80, Makespan[:, 4], color='k', width=0.20, label="HBWBA-AECC")
    plt.ylabel('Makespan (s)')
    plt.xlabel('Configuration')
    plt.xticks(X + 0.20, (10, 20, 30, 40, 50))
    plt.legend(loc='best')
    path1 = './Results/makespan_bar.png'
    plt.savefig(path1)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(0, 10, 2)
    Cost = Makespan / 200
    ax.bar(X + 0.00, Cost[:, 0], color='r', width=0.20, label="JA-AECC")
    ax.bar(X + 0.20, Cost[:, 1], color='g', width=0.20, label="MFO-AECC")
    ax.bar(X + 0.40, Cost[:, 2], color='b', width=0.20, label="BWO-AECC")
    ax.bar(X + 0.60, Cost[:, 3], color='m', width=0.20, label="BA-AECC")
    ax.bar(X + 0.80, Cost[:, 4], color='k', width=0.20, label="HBWBA-AECC")
    plt.ylabel('Cost (USD)')
    plt.xlabel('Configuration')
    plt.xticks(X + 0.20, (10, 20, 30, 40, 50))
    plt.legend(loc='best')
    path1 = './Results/Cost_bar.png'
    plt.savefig(path1)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(0, 10, 2)
    ax.bar(X + 0.00, Energy[:, 0], color='r', width=0.20, label="JA-AECC")
    ax.bar(X + 0.20, Energy[:, 1], color='g', width=0.20, label="MFO-AECC")
    ax.bar(X + 0.40, Energy[:, 2], color='b', width=0.20, label="BWO-AECC")
    ax.bar(X + 0.60, Energy[:, 3], color='m', width=0.20, label="BA-AECC")
    ax.bar(X + 0.80, Energy[:, 4], color='k', width=0.20, label="HBWBA-AECC")
    plt.ylabel('Energy')
    plt.xlabel('Configuration')
    plt.xticks(X + 0.20, (10, 20, 30, 40, 50))
    plt.legend(loc='best')
    path1 = './Results/Energy_bar.png'
    plt.savefig(path1)
    plt.show()

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    X = np.arange(0, 10, 2)
    ax.bar(X + 0.00, Res_Time[:, 0], color='r', width=0.20, label="JA-AECC")
    ax.bar(X + 0.20, Res_Time[:, 1], color='g', width=0.20, label="MFO-AECC")
    ax.bar(X + 0.40, Res_Time[:, 2], color='b', width=0.20, label="BWO-AECC")
    ax.bar(X + 0.60, Res_Time[:, 3], color='m', width=0.20, label="BA-AECC")
    ax.bar(X + 0.80, Res_Time[:, 4], color='k', width=0.20, label="HBWBA-AECC")
    plt.ylabel('Task Completion Time (s)')
    plt.xlabel('Configuration')
    plt.xticks(X + 0.20, (10, 20, 30, 40, 50))
    plt.legend(loc='best')
    path1 = './Results/Time_bar.png'
    plt.savefig(path1)
    plt.show()

    for i in range(Time.shape[0]):
        Graph_Time = Time[i, :, :]
        length = np.arange(8)
        plt.plot(length, Graph_Time[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="JA-AECC")
        plt.plot(length, Graph_Time[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="MFO-AECC")
        plt.plot(length, Graph_Time[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="BWO-AECC")
        plt.plot(length, Graph_Time[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="BA-AECC")
        plt.plot(length, Graph_Time[:, 4], color='c', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
                 label="HBWBA-AECC")
        plt.ylabel('Estimated Time (s)')
        plt.xlabel('No of Bits')
        plt.xticks(length, (20, 40, 60, 80, 100, 120, 140, 160))
        plt.legend(loc=4)
        plt.savefig('./Results/time_curve_' + str(i + 1) + 'line.png')
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(8)
        ax.bar(X + 0.00, Graph_Time[:, 5], color='r', width=0.20, label="RSA")
        ax.bar(X + 0.20, Graph_Time[:, 6], color='g', width=0.20, label="AES")
        ax.bar(X + 0.40, Graph_Time[:, 7], color='b', width=0.20, label="ECC")
        ax.bar(X + 0.60, Graph_Time[:, 8], color='m', width=0.20, label="HBWBA-AECC")
        plt.ylabel('Estimated Time (s)')
        plt.xlabel('No of Bits')
        plt.xticks(X + 0.20, (20, 40, 60, 80, 100, 120, 140, 160))
        plt.legend(loc='best')
        path1 = './Results/time_curve_' + str(i + 1) + 'bar.png'
        plt.savefig(path1)
        plt.show()

    for i in range(Size.shape[0]):
        Graph_Time = Size[i, :, :]
        length = np.arange(8)
        plt.plot(length, Graph_Time[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                 label="JA-AECC")
        plt.plot(length, Graph_Time[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label="MFO-AECC")
        plt.plot(length, Graph_Time[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                 label="BWO-AECC")
        plt.plot(length, Graph_Time[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                 label="BA-AECC")
        plt.plot(length, Graph_Time[:, 4], color='c', linewidth=3, marker='o', markerfacecolor='black', markersize=12,
                 label="HBWBA-AECC")
        plt.ylabel('Space Complexity')
        plt.xlabel('No of Bits')
        plt.xticks(length, (20, 40, 60, 80, 100, 120, 140, 160))
        plt.legend(loc=4)
        plt.savefig('./Results/size_curve_' + str(i + 1) + 'line.png')
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(8)
        ax.bar(X + 0.00, Graph_Time[:, 5], color='r', width=0.20, label="RSA")
        ax.bar(X + 0.20, Graph_Time[:, 6], color='g', width=0.20, label="AES")
        ax.bar(X + 0.40, Graph_Time[:, 7], color='b', width=0.20, label="ECC")
        ax.bar(X + 0.60, Graph_Time[:, 8], color='m', width=0.20, label="HBWBA-AECC")
        plt.ylabel('Space Complexity')
        plt.xlabel('No of Bits')
        plt.xticks(X + 0.10, (20, 40, 60, 80, 100, 120, 140, 160))
        plt.legend(loc='best')
        path1 = './Results/size_curve_' + str(i + 1) + 'bar.png'
        plt.savefig(path1)
        plt.show()


if __name__ == '__main__':
    Plot_Results()
