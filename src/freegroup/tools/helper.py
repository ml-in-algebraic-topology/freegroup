def value_or_default(value, default):
    return value if not value is None else default


def remove_prefix(prefix, kwargs):
    matching_keys = list(filter(lambda x: x.startswith(prefix), kwargs.keys()))
    return { k[len(prefix) + 1:]: kwargs.pop(k) for k in matching_keys }
