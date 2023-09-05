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
    sourceFile = 'DataSet/NSL-KDD/test.txt'
    # 处理完毕文件地址
    handledFile = 'DataSet/NSL-KDD/testHandled.cvs'
    # 分别记录每个特征的最大值和最小值
    minList = np.full(163,0)
    maxList = np.full(163, 0)
    with (open(sourceFile,'r')) as data_from,\
            (open(handledFile, 'w',newline='')) as data_to_flie:
        # 读文件和写文件
        csv_reader=csv.reader(data_from)
        csv_writer=csv.writer(data_to_flie)
        for line in csv_reader:
            temp_line=np.array(line)

            lenPro,resultPro = handleProtocol(line) #将源文件行中3种协议类型转换成数字标识
            lenSer,resultSer = handleService(line) #将源文件行中70种网络服务类型转换成数字标识
            lenFlag,resultFlag = handleFlag(line) #将源文件行中11种网络连接状态转换成数字标识
            lenLabel,resultLabel = handleLabel(line) #将源文件行中23种攻击类型转换成数字标识

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
            temp_line = np.insert(temp_line, indexLabel, resultLabel)

            # 找到极值
            for index in range(len(temp_line)):
                minList[index] = min(float(minList[index]),float(temp_line[index]))
                maxList[index] = max(float(maxList[index]),float(temp_line[index]))
            csv_writer.writerow(temp_line)
    print("数值化字符型特征完毕")
    return minList,maxList



# 最大最小值归一化处理
def minMaxHandle(minList,maxList):
    print("==================开始归一化处理================")
    handledFile = 'DataSet/NSL-KDD/testHandled.cvs'
    # 处理完毕文件临时地址
    tempFile = 'DataSet/NSL-KDD/KDDTrainTempHandled.cvs'
    with (open(handledFile,'r')) as data_from,\
            (open(tempFile, 'w',newline='')) as data_to_flie:
        # 读文件和写文件
        csv_reader=csv.reader(data_from)
        csv_writer=csv.writer(data_to_flie)
        for line in csv_reader:
            temp_line=np.array(line)
            # 全部归一化处理
            for index in range(len(temp_line)):
                temp_line[index] = normalizeX(float(temp_line[index]),float(minList[index]),float(maxList[index]))
            csv_writer.writerow(temp_line)
    print("归一化处理完毕")
    # 替换原始文件
    os.remove(handledFile)  # 删除原始文件
    os.rename(tempFile, handledFile)  # 将临时文件重命名为原始文件名


# 读取CSV文件并将数据转化为图像
def csvToImage():
    print("==================开始图像转化================")
    handledFile = 'DataSet/NSL-KDD/KDDTrainHandled.cvs'
    with open(handledFile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line,row in enumerate(reader, start=1):
            if line % 20000 == 0:
                print("处理完毕 %s 张图片" % line)
            # 将CSV行数据转换为一维矩阵
            rowData = [float(value) for value in row]
            # 开方向上取整
            shape = math.ceil(math.sqrt(len(rowData)))
            # 未满数据填充0
            rowData = rowData + [0] * (shape*shape-len(rowData))
            # 构建方阵，将列表转化为NumPy数组，并重塑为7x7的方阵
            matrix = np.array(rowData).reshape(shape, shape)
            # 构建图片
            image = Image.fromarray(matrix * 255)
            # 转换图像模式为RGBA
            image = image.convert('RGBA')
            image.save("C:/Users/29973/Desktop/论文/深度强化学习/论文复现/Image/"+str(line)+".png",'PNG')
    print("图片处理完毕")



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
    staut_list=['normal','ipsweep','mscan','nmap','portsweep','saint','satan',
                'apache2','back','land','mailbomb','neptune','pod','processtable',
                'smurf','teardrop','udpstorm','buffer_overflow','httptunnel','loadmodule',
                'perl','ps','rootkit','sqlattack','xterm','ftp_write','guess_passwd','imap',
                'multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy',
                'warezclient','warezmaster','worm','xlock','xsnoop']
    if input[41] in staut_list:
        return find_index(input[41],staut_list)
    else:
        staut_list.append(input[41])
        return find_index(input[41],staut_list)



# # 数值化字符型特征
# minList,maxList = oneHotHandle()
# # # 归一化处理
# minMaxHandle(minList, maxList)
# 转化为张量，进而转化为图片
csvToImage()




