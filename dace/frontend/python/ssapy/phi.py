

# Implementation Idea
# Phi returns:
#  if only 1 actual value supplied -> that value
#  if > 1 values supplied -> the last value
# this could (?) work if all phi nodes are constructed
# such that operands are ordered the way they are created

# Works for:
# If    phi's -> only one operand is defined
# While phi's -> if __phi__(before_ops, body_ops)
# For   phi's -> same as While

# Doesn't work for break-operands :(


def __phi__(*operands):
    ...