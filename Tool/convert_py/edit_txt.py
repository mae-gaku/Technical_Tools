import os

files = []
# Add the path of txt folder
for i in os.listdir(""):
    print(i)
    if i.endswith('.txt'):
        files.append(i)

print("files:",files)
for item in files:
    # define an empty list
    file_data = []

    # open file and read the content in a list
    with open(item, 'r') as myfile:
        for line in myfile:
            # remove linebreak which is the last character of the string
            currentLine = line[:-1]
            data = currentLine.split(" ")
            # add item to the list
            file_data.append(data)
    print("file_data",file_data)
    # Decrease the first number in any line by one
    for i in file_data:
        # print("i",i)
        if i[0].isdigit():
            # temp = 0
            # i[0] = str(int(temp))
            if i[0] == '2':
                temp = 1
                i[0] = str(int(temp))
            elif i[0] == '3':
                temp = 2
                i[0] = str(int(temp))
            elif i[0] == '4':
                temp = 3
                i[0] = str(int(temp))
            # elif i[0] == '3':
            #     temp = 1
            #     i[0] = str(int(temp))
            else:
                pass
                # print("pass")
    # print("file_data",file_data)

    # Write back to the file
    f = open(item, 'w')
    for i in file_data:
        res = ""
        for j in i:
            res += j + " "
        f.write(res)
        f.write("\n")
    f.close()