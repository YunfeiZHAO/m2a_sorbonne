
# def encryptionValidity(instructionCount, validityPeriod, keys):
#     # Write your code here
#     compute_power = instructionCount * validityPeriod
#     num_divider = 0
#     for key in keys:
#         num = 0
#         for k in keys:
#             if key % k == 0:
#                 num += 1
#         num_divider = max(num_divider, num)
#     strength = num_divider * (10 ** 5)
#     can = (compute_power > strength) * 1
#     return [can, strength]

def encryptionValidity(instructionCount, validityPeriod, keys):
    # Write your code here
    compute_power = instructionCount * validityPeriod
    key_dn = {}
    for key in keys:
        d = 0
        for k, v in key_dn.items():
            if k % key == 0:
                key_dn[k] += 1
            if key % k == 0:
                d += 1
        if key not in key_dn:
            key_dn[key] = d 

                
    num_divider = key_dn[max(key_dn)]
    strength = num_divider * (10 ** 5)
    can = (compute_power > strength) * 1
    return [can, strength]

if __name__ == "__main__":
    instructionCount = 100
    validityPeriod = 1000
    keys = [2,4]
    encryptionValidity(instructionCount, validityPeriod, keys)