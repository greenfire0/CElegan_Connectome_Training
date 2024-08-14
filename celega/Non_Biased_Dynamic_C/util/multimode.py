import collections

def custom_multimode(data):
    """Returns the most common data points from discrete or nominal data."""
    if not data:
        return []
    # Count the occurrences of each item in data
    counts = collections.Counter(data)
    # Find the maximum count
    max_count = max(counts.values())
    # Return all items that have the maximum count
    return [item for item, count in counts.items() if count == max_count]