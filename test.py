


a = [2,1,3,5,4,8]

min = a[0]
sec_min = a[0]

for i in range(len(a)):
    if a[i] < min:
        sec_min = min
        min = a[i]
    elif a[i] < sec_min and a[i] != min:
        sec_min = a[i]

print("the second min number is {}".format(sec_min))