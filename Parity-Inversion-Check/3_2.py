import heapq
import copy

# 定义目标状态
goal_state = [[1, 2, 3], [8, 0, 4], [7, 6, 5]]

# 定义操作算子集
operators = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 上、下、右、左

# 定义启发式函数（逆序对数）
def heuristic(state):
    flat_state = [value for row in state for value in row if value != 0]
    inversions = 0
    for i in range(len(flat_state)):
        for j in range(i + 1, len(flat_state)):
            if flat_state[i] > flat_state[j]:
                inversions += 1
    return inversions

# 定义状态节点类
class StateNode:
    def __init__(self, state, g_cost, parent):
        self.state = state  # 当前状态
        self.g_cost = g_cost  # 从初始状态到当前状态的实际代价
        self.h_cost = heuristic(state)  # 启发式函数估计的代价
        self.f_cost = self.g_cost + self.h_cost  # 总代价
        self.parent = parent  # 父节点

    def __lt__(self, other):
        return self.f_cost < other.f_cost

# 定义比较函数，用于优先队列
def compare_nodes(node1, node2):
    return node1.f_cost - node2.f_cost

# A*算法
def astar(initial_state):
    open_list = []  # 优先队列，用于存储待扩展的节点
    closed_set = set()  # 存储已经扩展过的节点

    # 初始化初始节点
    initial_node = StateNode(initial_state, 0, None)
    heapq.heappush(open_list, initial_node)

    expanded_nodes = 0
    generated_nodes = 1

    while open_list:
        current_node = heapq.heappop(open_list)
        expanded_nodes += 1

        if current_node.state == goal_state:
            # 找到解，输出路径
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path.reverse()

            return expanded_nodes, generated_nodes, path, open_list, closed_set

        closed_set.add(tuple(map(tuple, current_node.state)))

        for operator in operators:
            new_state = apply_operator(current_node.state, operator)
            if tuple(map(tuple, new_state)) not in closed_set:
                new_g_cost = current_node.g_cost + 1
                new_node = StateNode(new_state, new_g_cost, current_node)
                heapq.heappush(open_list, new_node)
                generated_nodes += 1

        if generated_nodes % 1000 == 0:
            print(f"Generated nodes: {generated_nodes}")

    return expanded_nodes, generated_nodes, None, open_list, closed_set

# 定义应用操作算子的函数
def apply_operator(state, operator):
    new_state = copy.deepcopy(state)
    zero_row, zero_col = find_zero(new_state)

    new_row = zero_row + operator[0]
    new_col = zero_col + operator[1]

    if 0 <= new_row < 3 and 0 <= new_col < 3:
        new_state[zero_row][zero_col], new_state[new_row][new_col] = (
            new_state[new_row][new_col],
            new_state[zero_row][zero_col],
        )

    return new_state

# 定义找到0所在位置的函数
def find_zero(state):
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                return i, j

# 主程序
initial_state = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]
expanded_nodes, generated_nodes, solution_path, open_list, closed_set = astar(initial_state)

print("Expanded nodes:", expanded_nodes)
print("Generated nodes:", generated_nodes)

if solution_path:
    print("\nSolution path:")
    for step, state in enumerate(solution_path):
        print(f"Step {step}:\n{state}\n")

    print("\nOpen table (first 5 states):")
    for i in range(min(5, len(open_list))):
        print(open_list[i].state)

    print("\nClosed table (first 5 states):")
    for i, state in enumerate(closed_set):
        if i >= 5:
            break
        print(state)
else:
    print("No solution found.")
