import csv

def writeCSV_AVG(csvFile, data):
    """
    Writes indexes to index database.
    ARGUMENTS:
        csvFile: filename
        data = [Filename, avg]
    """
    with open(csvFile, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(data)

def readCSV_AVG(csvFile):
    """
    Reads indexes from index database.
    RETURN:
    [Filename, avg]
    """
    result = []
    with open(csvFile, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in spamreader:
            result.append(row)
    for i in range(1,4):
        result[0][i] = float(result[0][i])
    return result[0]

# ********************** TESTING *********************** #

data = ['default/2.jpg', 73.67599487304685, 108.76373291015615, 72.97486368815103]
writeCSV_AVG('eggs.csv', data)
print(readCSV_AVG('eggs.csv'))

