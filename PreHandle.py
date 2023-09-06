'''
@Author ：YZX
@Date ：2023/8/31 15:53 
@Python-Version ：3.8
'''
#数据预处理
import numpy as np
import csv
import os
import math
from PIL import Image


# 数值化字符型特征
def oneHotHandle():
    print("==================开始数值化字符特征值================")
    # 源文件地址
    sourceFile = 'DataSet/NSL-KDD/KDDTrain+.txt'
    # 处理完毕文件地址
    handledFile = 'DataSet/NSL-KDD/KDDTrainHandled.cvs'
    # 分别记录每个特征的最大值和最小值
    minList = np.full(164,0)
    maxList = np.full(164, 0)
    with (open(sourceFile,'r')) as data_from,\
            (open(handledFile, 'w',newline='')) as data_to_flie:
        # 读文件和写文件
        csv_reader=csv.reader(data_from)
        csv_writer=csv.writer(data_to_flie)
        for count, line in enumerate(csv_reader, start=1):
            if count % 20000 == 0:
                print("数值化字符型特征处理完毕 %s 条数据" % count)

            temp_line=np.array(line)

            lenPro,resultPro = handleProtocol(line) #将源文件行中3种协议类型转换成数字标识
            lenSer,resultSer = handleService(line) #将源文件行中70种网络服务类型转换成数字标识
            lenFlag,resultFlag = handleFlag(line) #将源文件行中11种网络连接状态转换成数字标识
            category,resultListLabel = handleLabel(line) #将源文件行中23种攻击类型转换成数字标识,同时记录类别

            # 分别插入对应的位置
            temp_line = np.delete(temp_line,1)
            temp_line = np.insert(temp_line, 1,resultPro)

            indexSer = 1 + lenPro
            temp_line = np.delete(temp_line,indexSer)
            temp_line = np.insert(temp_line, indexSer,resultSer)

            indexFlag = indexSer + lenSer
            temp_line = np.delete(temp_line, indexFlag)
            temp_line = np.insert(temp_line, indexFlag, resultFlag)

            indexLabel = len(temp_line)-2
            temp_line = np.delete(temp_line, indexLabel)
            temp_line = np.insert(temp_line, indexLabel, resultListLabel[1])

            # 在表格头添加数据类别
            temp_line = np.insert(temp_line, 0, category)

            # 找到极值
            for index in range(len(temp_line)):
                # 排除掉第一个标签位置
                if index > 0:
                    minList[index] = min(float(minList[index]),float(temp_line[index]))
                    maxList[index] = max(float(maxList[index]),float(temp_line[index]))
            csv_writer.writerow(temp_line)
    print("数值化字符型特征完毕")
    return minList,maxList


# 最大最小值归一化处理
def minMaxHandle(minList,maxList):
    print("==================开始归一化处理================")
    handledFile = 'DataSet/NSL-KDD/KDDTrainHandled.cvs'
    # 处理完毕文件临时地址
    normalFile = 'DataSet/Change/Train/Normal.cvs'
    probeFile = 'DataSet/Change/Train/Probe.cvs'
    dosFile = 'DataSet/Change/Train/Dos.cvs'
    u2rFile = 'DataSet/Change/Train/U2R.cvs'
    r2lFile = 'DataSet/Change/Train/R2L.cvs'
    with (open(handledFile,'r')) as data_from,\
            (open(normalFile, 'w',newline='')) as normalFlie, \
            (open(probeFile, 'w', newline='')) as probeFlie, \
            (open(dosFile, 'w', newline='')) as dosFile,\
            (open(u2rFile, 'w',newline='')) as u2rFile, \
            (open(r2lFile, 'w', newline='')) as r2lFile:
        # 读文件和写文件
        csv_reader=csv.reader(data_from)
        normalWriter=csv.writer(normalFlie)
        probeWriter = csv.writer(probeFlie)
        dosWriter = csv.writer(dosFile)
        u2rWriter = csv.writer(u2rFile)
        r2lWriter = csv.writer(r2lFile)
        for count, line in enumerate(csv_reader, start=1):
            if count % 20000 == 0:
                print("归一化处理完毕 %s 条数据" % count)

            temp_line=np.array(line)
            # 全部归一化处理
            for index in range(len(temp_line)):
                # 排除掉第一个标签位置
                if index > 0:
                    temp_line[index] = normalizeX(float(temp_line[index]),float(minList[index]),float(maxList[index]))
            if temp_line[0] == 'Normal':
                normalWriter.writerow(np.delete(temp_line, 0))
            elif temp_line[0] == 'Probe':
                probeWriter.writerow(np.delete(temp_line, 0))
            elif temp_line[0] == 'Dos':
                dosWriter.writerow(np.delete(temp_line, 0))
            elif temp_line[0] == 'U2R':
                u2rWriter.writerow(np.delete(temp_line, 0))
            elif temp_line[0] == 'R2L':
                r2lWriter.writerow(np.delete(temp_line, 0))
    print("归一化处理完毕")


# 读取CSV文件并将数据转化为图像
def csvToImage():
    print("==================开始图像转化================")
    fileNameList = ['DataSet/Change/Train/Normal.cvs','DataSet/Change/Train/Probe.cvs',
                    'DataSet/Change/Train/Dos.cvs','DataSet/Change/Train/U2R.cvs',
                    'DataSet/Change/Train/R2L.cvs']
    for handledFile in fileNameList:
        # 获取文件夹名称
        name = os.path.splitext(os.path.basename(handledFile))[0]
        with open(handledFile, 'r') as csvfile:
            reader = csv.reader(csvfile)
            # 用来记录文件总共多少行
            sumCount = 0
            for count,line in enumerate(reader, start=1):
                sumCount = count
                if count % 10000 == 0:
                    print("处理完毕 %s 的 %s 张图片" % (name,count))
                # 将CSV行数据转换为一维矩阵
                rowData = [float(value) for value in line]
                # 开方向上取整
                shape = math.ceil(math.sqrt(len(rowData)))
                # 未满数据填充0
                rowData = rowData + [0] * (shape*shape-len(rowData))
                # 构建方阵，将列表转化为NumPy数组，并重塑为13x13的方阵
                matrix = np.array(rowData).reshape(shape, shape)
                # 构建图片
                image = Image.fromarray(matrix * 255)
                # 转换图像模式为RGBA
                image = image.convert('RGBA')
                image.save("C:/Users/29973/Desktop/论文/深度强化学习/论文复现/TrainImage/"+name+"/"+str(count)+".png",'PNG')
            print("===处理完毕 %s 的共 %s 张图片===" % (name, sumCount))


# 归一化数据
def normalizeX(x, minX, maxX):
    if minX == maxX:
        return 0
    return (x-minX)/(maxX-minX)


# 按照提供的列表，返回当前输入字符在列表的索引位置，并在前或后补零
def find_index(searchWord, searchList):
    resultList = np.full(len(searchList), 0)
    index = [a for a in range(len(searchList)) if searchList[a] == searchWord]
    resultList[index] = 1
    return len(resultList),resultList

# 总共3种协议值
def handleProtocol(input):
    protoclo_list=['tcp','udp','icmp']
    if input[1] in protoclo_list:
        return find_index(input[1],protoclo_list)


# 总共70种服务模式
def handleService(input):
    service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard',
                  'domain','domain_u','echo','eco_i','ecr_i','efs','exec','finger',
                  'ftp','ftp_data','gopher','harvest','hostnames','http','http_2784',
                  'http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell',
                  'ldap','link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn',
                  'netstat','nnsp','nntp','ntp_u','other','pm_dump','pop_2','pop_3','printer',
                  'private','red_i','remote_job','rje','shell','smtp','sql_net','ssh',
                  'sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i',
                  'urp_i','uucp','uucp_path','vmnet','whois','X11','Z39_50']
    if input[2] in service_list:
        return find_index(input[2],service_list)

# 总共11种标签
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)


# 总共40种访问方法，包括1个正常+39种攻击
def handleLabel(input):
    # 大分类
    staut_list=['normal','ipsweep','mscan','nmap','portsweep','saint','satan',
                'apache2','back','land','mailbomb','neptune','pod','processtable',
                'smurf','teardrop','udpstorm','buffer_overflow','httptunnel','loadmodule',
                'perl','ps','rootkit','sqlattack','xterm','ftp_write','guess_passwd','imap',
                'multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy',
                'warezclient','warezmaster','worm','xlock','xsnoop']
    # 小分类
    probe_list = ['ipsweep','mscan','nmap','portsweep','saint']
    dos_list = ['apache2','back','land','mailbomb','neptune','pod','processtable','smurf','teardrop','udpstorm']
    u2r_list = ['buffer_overflow','httptunnel','loadmodule','perl','ps','rootkit','sqlattack','xterm']

    if input[41] in staut_list:
        # 还需要进行归类处理
        if input[41] == 'normal':
            return 'Normal',find_index(input[41],staut_list)
        else:
            if input[41] in probe_list:
                return 'Probe',find_index(input[41],staut_list)
            elif input[41] in dos_list:
                return 'Dos',find_index(input[41],staut_list)
            elif input[41] in u2r_list:
                return 'U2R',find_index(input[41],staut_list)
            else:
                return 'R2L',find_index(input[41],staut_list)
    else:
        staut_list.append(input[41])
        return 'None',find_index(input[41],staut_list)



def test():
    file = 'DataSet/Change/Train/Normal.cvs'
    name = os.path.splitext(os.path.basename(file))[0]
    result = "C:/Users/29973/Desktop/论文/深度强化学习/论文复现/TrainImage/" + name + "/" + str(1) + ".png"
    print(result)

# # 数值化字符型特征
# minList,maxList = oneHotHandle()
# # 归一化处理
# minMaxHandle(minList, maxList)
# 转化为张量，进而转化为图片
csvToImage()





