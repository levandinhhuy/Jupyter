# Solve the following MIP:
#  maximize
#        12 x + 14 y
#  subject to
#        1.5 x + 2 y <= 18
#        12 x + 10 y <= 120
#        x, y >= 0 && x, y are integers

import gurobipy as gp
from gurobipy import GRB

# Tạo model
m = gp.Model("mip_example")

# Tạo biến nguyên không âm
x = m.addVar(vtype=GRB.INTEGER, name="x")
y = m.addVar(vtype=GRB.INTEGER, name="y")

# Đặt hàm mục tiêu: maximize 12x + 14y
m.setObjective(12 * x + 14 * y, GRB.MAXIMIZE)

# Thêm ràng buộc
m.addConstr(1.5 * x + 2 * y <= 18, name="c1")
m.addConstr(12 * x + 10 * y <= 120, name="c2")

# Giải mô hình
m.optimize()

# In kết quả
if m.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {m.objVal}")
    print(f"x = {x.X}, y = {y.X}")
