class RuleBasedMemoryPolicy:
    """
    Simple heuristic-based memory policy.
    """

    def __init__(self, max_memory_size=5, importance_threshold=0.6):
        self.max_memory_size = max_memory_size
        self.importance_threshold = importance_threshold

    def select_action(self, observation, memory):
        """
        observation = [item_importance, memory_size, avg_memory_importance]
        """
        item_importance = observation[0]

        # If item is important and memory has space, store
        if item_importance >= self.importance_threshold:
            if len(memory) < self.max_memory_size:
                return 1  # STORE
            else:
                # If memory full, check if new item is better than worst
                if min(memory) < item_importance:
                    return 2  # DELETE OLDEST (we will delete manually)
                else:
                    return 0  # IGNORE

        return 0  # IGNORE
