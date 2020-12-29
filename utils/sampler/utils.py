def make_weights_for_balanced_classes(data, nclasses):
    classes_cnt = [0] * nclasses
    weight_per_class = [0.] * nclasses
    weight_for_all_images = [0.] * len(data)

    for item in data:
        classes_cnt[item[1]] += 1   # clw note: data由很多item组成，每个item是图片和label构成的列表; item[1]即 label
    N = float(sum(classes_cnt))  # clw note: image_nums
    for i in range(nclasses):
        weight_per_class[i] = N / float(classes_cnt[i])
    for idx, item in enumerate(data):
        weight_for_all_images[idx] = weight_per_class[item[1]]
    return weight_for_all_images