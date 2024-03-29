import os
from util import *
import numpy as np
import multiprocessing

train_hex = {2, 5, 10, 12, 15, 17, 24, 31, 40, 44, 47, 50, 52, 54, 69, 77, 79, 83, 90, 94, 117, 129, 134, 158, 167, 171,
             174, 179, 185, 189, 193, 204, 210, 211, 215, 235, 239, 255, 279, 281, 290, 291, 293, 307, 321, 333, 336,
             337, 338, 339, 356, 358, 378, 379, 382, 383, 398, 404, 415, 422, 436, 447, 449, 452, 454, 456, 469, 478,
             481, 499, 514, 522, 551, 562, 563, 566, 577, 591, 594, 599, 606, 611, 612, 637, 639, 640, 642, 644, 647,
             661, 666, 667, 675, 680, 683, 684, 689, 697, 698, 706, 713, 719, 723, 731, 736, 738, 745, 751, 755, 762,
             763, 774, 778, 781, 809, 811, 813, 815, 827, 828, 833, 840, 852, 854, 863, 871, 879, 904, 915, 922, 924,
             925, 944, 953, 958, 975, 985, 986, 1001, 1002, 1016, 1019, 1020, 1044, 1049, 1052, 1053, 1063, 1067, 1068,
             1070, 1093, 1094, 1095, 1096, 1100, 1101, 1106, 1109, 1129, 1134, 1138, 1176, 1178, 1186, 1202, 1206, 1218,
             1226, 1235, 1236, 1250, 1262, 1271, 1284, 1300, 1302, 1308, 1330, 1332, 1335, 1347, 1357, 1358, 1362, 1380,
             1382, 1387, 1407, 1408, 1411, 1412, 1413, 1415, 1416, 1434, 1437, 1438, 1442, 1457, 1465, 1493, 1497, 1499,
             1528, 1529, 1536, 1552, 1554, 1555, 1559, 1566, 1576, 1590, 1592, 1597, 1608, 1619, 1630, 1638, 1649, 1653,
             1657, 1660, 1661, 1671, 1676, 1677, 1680, 1701, 1706, 1713, 1715, 1721, 1727, 1731, 1735, 1742, 1743, 1744,
             1750, 1751, 1758, 1768, 1770, 1772, 1777, 1778, 1782, 1788, 1789, 1793, 1795, 1803, 1811, 1813, 1821, 1829,
             1831, 1844, 1867, 1874, 1877, 1880, 1882, 1898, 1909, 1915, 1925, 1928, 1931, 1933, 1936, 1938, 1939, 1943,
             1948, 1955, 1958, 1976, 1981, 1984, 1985, 1987, 1993, 2004, 2011, 2012, 2015, 2022, 2033, 2051, 2054, 2062,
             2066, 2068, 2074, 2088, 2091, 2093, 2095, 2096, 2099, 2111, 2112, 2115, 2131, 2132, 2138, 2139, 2153, 2157,
             2162, 2172, 2174, 2178, 2180, 2185, 2200, 2203, 2206, 2207, 2209, 2211, 2219, 2225, 2232, 2240, 2243, 2249,
             2250, 2268, 2274, 2276, 2281, 2284, 2304, 2305, 2316, 2322, 2326, 2327, 2331, 2348, 2350, 2357, 2362, 2363,
             2371, 2372, 2375, 2394, 2395}

train_fcc = {3,6,27,28,32,39,49,55,63,64,70,71,85,89,93,118,121,140,153,159,168,172,173,181,190,192,195,209,214,222,231,
    232,241,250,252,264,266,274,280,286,288,289,296,298,299,301,304,305,315,316,322,327,335,340,360,361,362,363,390,392,396,
    399,401,405,424,427,433,440,459,465,466,474,477,486,489,502,513,517,518,523,527,534,539,541,543,545,546,554,556,559,560,
    571,582,588,589,597,600,603,608,616,620,630,634,638,646,655,671,677,682,693,701,704,709,714,720,722,725,730,734,737,741,
    743,747,764,775,789,790,794,796,806,817,831,834,836,839,842,846,872,883,892,893,895,896,901,902,909,914,921,923,929,936,
    937,938,947,959,963,970,980,1012,1031,1034,1064,1076,1102,1108,1111,1116,1119,1120,1122,1126,1127,1135,1137,1140,1144,1158,
    1162,1170,1172,1173,1175,1180,1187,1193,1210,1213,1216,1217,1221,1227,1228,1232,1239,1241,1246,1255,1257,1258,1261,1265,1268,
    1275,1281,1286,1293,1298,1301,1303,1311,1316,1321,1329,1336,1338,1339,1341,1343,1345,1346,1350,1356,1360,1361,1365,1366,1367,
    1369,1373,1376,1384,1397,1398,1401,1403,1417,1420,1423,1425,1426,1427,1428,1431,1445,1454,1456,1460,1461,1481,1486,1487,1502,
    1506,1514,1526,1527,1534,1535,1539,1542,1546,1549,1550,1551,1564,1570,1571,1574,1575,1582,1586,1588,1594,1598,1604,1609,1612,
    1622,1636,1637,1645,1652,1658,1663,1670,1684,1689,1690,1691,1711,1714,1716,1717,1718,1720,1723,1732,1739,1741,1766,1767,1786,
    1787,1804,1805,1806,1832,1840,1842,1845,1847,1851,1862,1876,1879,1881,1890,1893,1921,1934,1959,1960,1964,1965,1969,2003,2010,
    2017,2020,2024,2032,2040,2045,2055,2057,2063,2064,2065,2067,2069,2073,2076,2098,2105,2108,2109,2116,2136,2137,2140,2142,2146,
    2155,2159,2168,2171,2175,2179,2182,2202,2213,2214,2216,2217,2241,2242,2246,2261,2266,2269,2273,2282,2289,2290,2319,2321,2324,
    2330,2333,2335,2337,2339,2341,2360,2364,2365,2370,2374,2380,2382,2386,2390,2392}

test_hex = {13, 16, 19, 33, 49, 54, 58, 63, 64, 65, 69, 70, 76, 90, 92, 96, 114, 115, 120, 128, 129, 143, 154, 166, 167,
            172, 183, 185, 186, 187, 191, 194, 195, 207, 209, 214, 219, 221, 223, 254, 264, 267, 272, 273, 275, 278,
            291, 293, 305, 316, 334, 346, 349, 350, 361, 366, 371, 373, 378, 381, 387, 392, 393, 407, 423, 434, 438,
            441, 443, 451, 455, 457, 468, 472, 474, 483, 486, 494, 496, 511, 515, 518, 530, 555, 557, 561, 567, 570,
            571, 598}

test_fcc = {14,45,55,56,60,61,77,104,105,111,117,124,130,132,141,145,148,162,164,165,174,177,178,184,189,203,208,211,
    217,226,227,228,236,245,247,255,259,271,274,284,290,292,294,295,304,308,310,312,319,324,382,384,386,395,398,400,404,
    411,416,418,421,425,426,431,439,442,449,458,461,467,469,479,491,495,502,505,507,508,509,529,541,544,545,548,551,554,
    565,568,576,577,589,590,595}

def augment_date(filename, repeats=10, fineness=1):
    sc = super_cell(filename)
    t0 = np.asarray([sc.to_tensor(fineness=fineness)])
    for _ in range(repeats):
        t0 = np.concatenate((t0,
                             np.asarray([
                                 sc.random_transform().to_tensor(fineness=fineness,
                                                                 gamma_al=1.0,
                                                                 gamma_ga=1.0,
                                                                 gamma_in=1.0,
                                                                 gamma_o=1.0)])))
    return t0


def augment_data2(filename, repeats=10, fineness=1, train=True):
    _, i, _ = filename.split('/')
    isHex = False
    if (int(i) in train_hex and train) or (int(i) in test_hex and not train):
        isHex = True
        print('{} is hexagonal'.format(i))
    sc = super_cell(filename, isHex)
    t0 = np.asarray([sc.to_tensor2(fineness=fineness, cell_size=13)])
    for _ in range(repeats):
        t0 = np.concatenate((t0,
                             np.asarray([
                                 sc.random_transform().to_tensor2(fineness=fineness, cell_size=13)])))
    return t0

def mk_aug_data(f):
    tensor = augment_data2('test/{}/geometry.xyz'.format(f))
    np.save('test/{}/tensor_aug2'.format(f), tensor)

def mk_aug_special_train(f):
    if int(f) in train_hex:
        tensor = augment_data2('train/{}/geometry.xyz'.format(f), repeats=40, train=True)
        np.save('train/{}/tensor_aug2'.format(f), tensor)
    elif int(f) in train_fcc:
        tensor = augment_data2('train/{}/geometry.xyz'.format(f), repeats=80, train=True)
        np.save('train/{}/tensor_aug2'.format(f), tensor)
    else:
        tensor = augment_data2('train/{}/geometry.xyz'.format(f), repeats=10, train=True)
        np.save('train/{}/tensor_aug2'.format(f), tensor)

def mk_aug_special_test(f):
    if int(f) in test_hex:
        tensor = augment_data2('test/{}/geometry.xyz'.format(f), repeats=40, train=False)
        np.save('test/{}/tensor_aug2'.format(f), tensor)
    elif int(f) in test_fcc:
        tensor = augment_data2('test/{}/geometry.xyz'.format(f), repeats=80, train=False)
        np.save('test/{}/tensor_aug2'.format(f), tensor)
    else:
        tensor = augment_data2('test/{}/geometry.xyz'.format(f), repeats=10, train=False)
        np.save('test/{}/tensor_aug2'.format(f), tensor)

if __name__ == '__main__':
    #mk_aug_special_train('3')
    pool = multiprocessing.Pool(7)
    pool.map(mk_aug_special_train, os.listdir('train'))
    pool.map(mk_aug_special_test, os.listdir('test'))