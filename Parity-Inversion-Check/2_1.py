import numpy as np
def getFunctionValue(self):
    # 使用曼哈顿距离作为估价函数
    cur_node = self.state.copy()
    fin_node = self.answer.copy()
    dist = 0
    N = len(cur_node)

    for i in range(N):
        for j in range(N):
            if cur_node[i][j] != fin_node[i][j] and cur_node[i][j] != 0:
                x, y = divmod(np.where(fin_node == cur_node[i][j]), N)
                dist += abs(x - i) + abs(y - j)

    return dist + self.depth