import functools
print("Task 1")

def mySum(x, y):
    return x+y

s = mySum(20, 21)
print(s)

print("Task 2")

def pascal(roww: int, column: int) -> int:
    def triangle(row: int) -> int:
        if row == 0:
            return [[1]]
        else:
            new_layer = [1]
            result = triangle(row-1)
            last_layer = result[-1]
            for i in range(len(last_layer)-1):
                new_layer.append(last_layer[i] + last_layer[i+1])
            new_layer += [1]
            result.append(new_layer)
        return result

    return triangle(roww)[roww][column]

print(pascal(4,2))

print("Task 3")
def balance(chars: list) -> bool:
    if len(chars) == 0:
        return True

    elif chars[0] == '¤':
        if len(chars) % 2 == 0:
            check = True
            return check
        else:
            check = False
            return check

    elif chars[0] == '(' or chars[0] == ')':
        chars.append('¤')
        chars.pop(0)
        balance(chars)

    else:
        chars.pop(0)
        balance(chars)

    if chars[0] == '¤' or len(chars) == 0:
        if len(chars) % 2 == 0:
            check = True
            return check
        else:
            check = False
            return check

print(balance(['(',')','a']))

print("Task 4")
a = [1, 2, 3, 4, 5]

b = list(map(lambda x: x**2, range(len(a)+1)))
b.pop(0)
c = functools.reduce(lambda x,y: x+y, b)

print(a, b, c)
