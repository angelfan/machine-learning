# coding: utf-8

def value_type_compare(obj1, obj2):
    return type(obj1) == type(obj2)

def diff(dict1, dict2, options=None):
    if options is None:
        options = {}
    opts = {"prefix": ""}
    opts.update(options)

    prefix = opts.get("prefix")
    if dict1 is None and dict2 is None:
        return []

    if dict1 is None:
        return [['~', prefix, None, dict2]]

    if dict2 is None:
        return [['~', prefix, dict1, None]]

    result = []

    if isinstance(dict1, dict):
        deleted_keys = set(dict1.keys())-set(dict2.keys())
        added_keys = set(dict2.keys())-set(dict1.keys())

        common_keys = set(dict1.keys()) & set(dict2.keys())

        if prefix == '':
            prefix = ''
        else:
            prefix = prefix + '.'

        for k in deleted_keys:
            result.append(['-', prefix + k, dict1.get(k), None])

        for k in sorted(added_keys):
            result.append(['+', prefix + k, None, dict2.get(k)])

        for k in common_keys:
            opts.update({"prefix": prefix + k})
            compare_result = diff(dict1.get(k), dict2.get(k), opts)
            if len(compare_result) != 0:
                result.extend(compare_result)
    else:
        if value_type_compare(dict1, dict2):
            return []
        else:
            return [['~', opts.get("prefix"), dict1, dict2]]

    return result

obj1 = {
    "ticker": {
        "high": "2894.97",
        "low": "2850.08",
        "buy": "2876.92",
        "sell": "2883.80",
        "last": "2875.66",
        "vol": "4133.63800000",
        "date": 1396412995,
        "vwap": 2879.12,
        "prev_close": {'a': 's'},
        "open": 2880.01
    }
}
obj2 = {
    "ticker": {
        "high": "2894.97",
        "low": "2850.08",
        "buy": "2876.92",
        "sell": "2883.80",
        "last": "2875.66",
        "vol": "4133.63800000",
        "date": 1396412995,
        "vwap": 2879.12,
        "prev_close": {'b': []},
        "open": '2880.02'
    }
}
print diff(obj1, obj2)


