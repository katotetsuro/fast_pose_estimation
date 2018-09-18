"""
from COCO
"""
num_keypoints = 17

"""
search window size for limbs
"""
Hp = 9
Wp = 9

"""
from
https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py#L49-L63
"""
num_limbs = 15

output_dim = 6 * (num_keypoints + 1) + Hp * Wp * num_limbs

stride = 32
input_width = 384.0
output_cols = input_width / stride
input_height = 384.0
output_rows = input_height / stride
