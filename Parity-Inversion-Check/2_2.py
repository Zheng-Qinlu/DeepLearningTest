def getFunctionValue(self):
    # 使用逆序对数作为估价函数
    cur_node = self.state.copy().flatten()
    fin_node = self.answer.copy().flatten()
    dist = 0

    for i in range(len(cur_node) - 1):
        for j in range(i + 1, len(cur_node)):
            if cur_node[i] > 0 and cur_node[j] > 0 and cur_node[i] > cur_node[j] and fin_node[i] > fin_node[j]:
                dist += 1

    return dist + self.depth