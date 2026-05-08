import sys

import numpy as np
import time


class State:
    def __init__(self, state, directionFlag=None, parent=None, f=0,depth=0):
        self.state = state
        self.direction = ['up', 'down', 'right', 'left']
        if directionFlag:
            self.direction.remove(directionFlag)
        self.parent = parent
        self.f = f
        self.depth=depth

    def getDirection(self):
        return self.direction

    def setF(self, f):
        self.f = f
        return

    # 打印结果
    def showInfo(self):
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                print(self.state[i, j], end='  ')
            print("\n")
        print('->')
        return

    # 获取0点
    def getZeroPos(self):
        postion = np.where(self.state == 0)
        return postion

    # 不在位节点数  f = g + h；另外的估价函数可以考虑用曼哈顿距离，逆序对数等
    def getFunctionValue(self):
        cur_node = self.state.copy()
        fin_node = self.answer.copy()
        dist = 0
        N = len(cur_node)

        for i in range(N):
            for j in range(N):
                if cur_node[i][j] != fin_node[i][j]:
                    dist+=1
        return dist + self.depth

    def nextStep(self):
        if not self.direction:
            return []
        subStates = []
        boarder = len(self.state) - 1
        # 获取0点位置
        x, y = self.getZeroPos()
        # 向左
        if 'left' in self.direction and y > 0:
            s = self.state.copy()
            tmp = s[x, y - 1]
            s[x, y - 1] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='right', parent=self,depth=self.depth+1)          #这里注意当前节点可以向左移动，那么向左移动产生的子节点应该不能再向右移动，否则变回原来的节点，因此要去掉向右的方向
            news.setF(news.getFunctionValue())
            subStates.append(news)
        # 向上
        if 'up' in self.direction and x > 0:
            # it can move to upper place
            s = self.state.copy()
            tmp = s[x - 1, y]
            s[x - 1, y] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='down', parent=self,depth=self.depth+1)
            news.setF(news.getFunctionValue())
            subStates.append(news)
        # 向下
        if 'down' in self.direction and x < boarder:
            # it can move to down place
            s = self.state.copy()
            tmp = s[x + 1, y]
            s[x + 1, y] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='up', parent=self,depth=self.depth+1)
            news.setF(news.getFunctionValue())
            subStates.append(news)
        # 向右
        if self.direction.count('right') and y < boarder:
            # it can move to right place
            s = self.state.copy()
            tmp = s[x, y + 1]
            s[x, y + 1] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='left', parent=self,depth=self.depth+1)
            news.setF(news.getFunctionValue())
            subStates.append(news)
        # 返回F值最小的下一个点
        #subStates.sort(key=compareNum)

        return subStates

    # A* 迭代
    def solve(self):
        # openList
        openTable = []
        # closeList
        closeTable = []
        openTable.append(self)

        while len(openTable) > 0:
            # 下一步的点移除open
            openTable.sort(key=compareNum)
            n = openTable.pop(0)
            # 加入close
            closeTable.append(n)
            # 确定下一步点
            nextstates=n.nextStep()
            openTable.extend(nextstates)

            #subStates = n.nextStep()
            for subStates in nextstates:
                path = []
                # 判断是否和最终结果相同
                if (subStates.state == subStates.answer).all():
                    while subStates.parent and subStates.parent != originState:
                        path.append(subStates.parent)
                        subStates = subStates.parent
                    path.reverse()
                    return path,openTable,closeTable
            #openTable.append(subStates)
        else:
            return None, None


def compareNum(state):
    return state.f


if __name__ == '__main__':
    originState = State(np.array([[2, 8, 3], [1, 6, 4], [7, 0, 5]]))
    State.answer = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])

    s1 = State(state=originState.state)
    path,openTable,closeTable = s1.solve()
    if path:
        for node in path:
            node.showInfo()
        print(State.answer)
        print("Total steps is %d" % len(path))