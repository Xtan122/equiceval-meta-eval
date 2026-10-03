"""Tên và ý nghĩa các contract đánh giá EquiCEval.

Giữ tên ở một nơi giúp evaluator, oracle relabel và CLI không vô tình dùng
những tập lựa chọn khác nhau.
"""

FEASIBLE_SET = "feasible_set"
OBJECTIVE_AFFINE = "feasible_set_and_objective_affine"
OBJECTIVE_VALUE = "feasible_set_and_objective_value"
ARGMIN = "feasible_set_and_argmin"

SUPPORTED_CONTRACTS = (
    FEASIBLE_SET,
    OBJECTIVE_AFFINE,
    OBJECTIVE_VALUE,
    ARGMIN,
)

# Contract chính cho nghiên cứu EquivaFormulation: giữ miền nghiệm và thứ tự
# ưu tiên toàn cục; cho phép đổi đơn vị/tỷ lệ dương và cộng hằng số.
PRIMARY_EQUIVAFORMULATION_CONTRACT = OBJECTIVE_AFFINE
