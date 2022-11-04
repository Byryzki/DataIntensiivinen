def pascal(row: int, column: int) -> int:
    if row == 0:
        return [[1]]
    else:
        new_layer = [1]
        result = pascal(row-1, column)
        last_layer = result[-1]
        for i in range(len(last_layer)-1):
            new_layer.append(last_layer[i] + last_layer[i+1])
        new_layer += [1]
        result.append(new_layer)
    return result

print(pascal(3,3))