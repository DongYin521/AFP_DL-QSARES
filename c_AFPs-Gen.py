import itertools
import pymysql
import pickle

# Establishing a database connection (When the peptide chain is long, file storage is not appropriate)
db1 = pymysql.connect(user='root', password='980521', host='localhost', database='ap', port=3306)

# Dominant amino acid (Dipeptide composition)
n1 = ('R', 'V', 'G', 'K')
n2 = ('R', 'L')
n3 = ('A', 'W', 'L')
n4 = 'R'
n5 = 'I'
n6 = ('V', 'R', 'I')
n7 = ('V', 'L', 'K')
n8 = 'I'
n9 = 'R'
n10 = ('I', 'V', 'R', 'K')
n11 = ('R', 'A', 'L')
n12 = ('R', 'K', 'A')  # Can expand longer peptide chains according to rules.

ls1 = itertools.product(n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12)  # Cartesian product
chargeDict = pickle.load(open('./pkl/chargeDict.pkl', 'rb'))
GravyDict = pickle.load(open('./pkl/GravyDict.pkl', 'rb'))
WimleyDict = pickle.load(open('./pkl/WimleyDict.pkl', 'rb'))
num = 0
for tuple1 in ls1:
    num = num + 1
    str2 = ''.join(tuple1)
    list1 = list(tuple1)

    chargeList = []  # Net Charge
    for i in list1:
        chargeList.append(chargeDict[i])

    GravyList = []  # GRAVY
    for j in list1:
        GravyList.append(GravyDict[j])

    WimleyList = []  # The Wimley-White whole-residue hydrophobicity of the peptide
    for m in list1:
        WimleyList.append(WimleyDict[m])

    length = len(str2)
    net_Charge = sum(chargeList)
    Gravy = sum(GravyList) / length
    Wimley = sum(WimleyList)
    cursor = db1.cursor()
    sql = "INSERT INTO afp_7776(sequence, length, charge, gravy, wimley, id) VALUES" \
          "('%s','%d','%d','%.2f','%.2f','%d')" % (str2, length, net_Charge, Gravy, Wimley, num)
    cursor.execute(sql)
    db1.commit()  # Commit to the database for execution
    print(str2, length, net_Charge, Gravy, Wimley, num)

cursor.close()
db1.close()
