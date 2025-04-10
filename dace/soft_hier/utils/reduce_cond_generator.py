class reduce_cond_generator:
    """
    This class generates the reduction condition for the soft hierarchical reduction
    """

    def __init__(self):
        pass

    def reduce_to_first(self, gi, gj, kg_i, kg_j, kg_m, kg_n, gM, gN):
        """
        Generate the reduction condition
        """
        kg_i = gi // kg_m
        kg_j = gj // kg_n
        kg_oi = gi % kg_m
        kg_oj = gj % kg_n
        kg_num = kg_m * kg_n
        kg_off = kg_oi * kg_n + kg_oj
        # Get the reduction condition from the node
        reduce_cond = f"{kg_off} == 0"

        # Get the reduction condition from the node
        return reduce_cond
    