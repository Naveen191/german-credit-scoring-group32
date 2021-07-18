f =  open("germanCreditData_hackathon.data")
l = []
for i in f:
    each = i.split()
    s = ""
    for j in range(len(each)):
        if j in [0,5,8,9,10,11,12,13,15,18]:
            pass
        else:
            s+=each[j]+","
    print(s[:-1])
    l.append(s[:-1]+"\n")
k = open("test.csv","a")
k.writelines(l)
