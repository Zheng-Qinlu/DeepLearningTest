import numpy as np

class EightPuzzleSolver:
    def __init__(self, initial_state, target_state):
        self.initial_state = np.array(initial_state).flatten()  # 将初始状态转换为一维数组
        self.target_state = np.array(target_state).flatten()    # 将目标状态转换为一维数组

    def count_inversions(self, arr):
        """
        计算逆序对数
        """
        inv_count = 0
        n = len(arr)

        for i in range(n - 1):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    inv_count += 1

        return inv_count

    def is_solvable(self):
        """
        判断八数码问题是否有解，并输出逆序对数
        """
        inv_initial = self.count_inversions(self.initial_state)
        inv_target = self.count_inversions(self.target_state)

        # 如果逆序对数奇偶性相同，则有解
        solvable = inv_initial % 2 == inv_target % 2

        return solvable, inv_initial, inv_target

if __name__ == '__main__':
    # 设置不同的初始状态和目标状态
    initial_state1 = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]
    target_state1 = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

    initial_state2 = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    target_state2 = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]

    # 创建解决器对象
    solver1 = EightPuzzleSolver(initial_state1, target_state1)
    solver2 = EightPuzzleSolver(initial_state2, target_state2)

    # 判断是否有解并输出逆序对数
    solvable1, inv_initial1, inv_target1 = solver1.is_solvable()
    solvable2, inv_initial2, inv_target2 = solver2.is_solvable()

    print("Problem 1 is solvable:", solvable1)
    print("Inversions in Problem 1 - Initial State:", inv_initial1, "Target State:", inv_target1)

    print("Problem 2 is solvable:", solvable2)
    print("Inversions in Problem 2 - Initial State:", inv_initial2, "Target State:", inv_target2)
