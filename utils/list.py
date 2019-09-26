

def value_index_max(ma_list):
    index_max, max =0, 0

    for i in range(len(ma_list)):
        if ma_list[i] > max :
            max = ma_list[i]
            index_max = i

    return index_max, max
