import datetime


def num_unique_pairs(arr):
    """
    Count the number of unique pairs in a list.

    Args:
        arr (list): List of elements.

    Returns:
        int: Number of unique pairs.
    """
    return len(arr) * (len(arr) - 1) // 2



def from_timestamp_to_date(timestamp):
    """
    Convert a timestamp to a date string.

    Args:
        timestamp (int): Timestamp in seconds.

    Returns:
        str: Date string in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')




if __name__ == "__main__":
    # Example usage
    arr = [1, 2, 3, 4]
    print(num_unique_pairs(arr))  # Output: 6

    timestamp = 1633072800
    print(from_timestamp_to_date(timestamp))  # Output: '2021-10-01 09:20:00'