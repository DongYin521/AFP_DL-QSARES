import pickle

F1 = open('./pkl/chargeDict.pkl', 'rb')
F2 = open('./pkl/GravyDict.pkl', 'rb')
F3 = open('./pkl/WimleyDict.pkl', 'rb')
content1 = pickle.load(F1)
content2 = pickle.load(F2)
content3 = pickle.load(F3)

print(content1, '\n', content2, '\n', content3)
