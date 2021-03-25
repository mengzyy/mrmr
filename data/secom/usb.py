def readdata():
    f = open("data/secom/secom.data")  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    featurelist = []
    lable = []
    while line:
        split = line.split(" ")
        temp = []
        for t in split:
            if t == 'NaN' or t=='nan':
                temp.append(0.0)
            else:
                temp.append(float(t))
        featurelist.append(temp)
        line = f.readline()
    f.close()

    f = open("data/secom/secom_labels.data")  # 返回一个文件对象
    line = f.readline()  # 调用文件的 readline()方法
    while line:
        split = line.split(" ")
        lable.append(int(split[0]))
        line = f.readline()
    f.close()
    return featurelist, lable

# print(readdata()[1])
