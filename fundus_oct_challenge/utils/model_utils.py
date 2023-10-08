def normalize(weights):
    total = sum(weights)
    if total == 0:  # avoid division by zero
        return [0] * len(weights)
    return [w / total for w in weights]