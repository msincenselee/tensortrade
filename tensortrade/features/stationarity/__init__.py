from .fractional_difference import FractionalDifference

# 根据环境十分稳定、可以将强化学习问题分为stationary、non-stationary。
#如果状态转移和奖励函数是确定的，即选择动作a后执行它的结果是确定的，那么这个环境就是stationary。
#如果状态转移或奖励函数是不确定的，即选择动作a后执行它的结果是不确定的，那么这个环境就是non-stationary。
