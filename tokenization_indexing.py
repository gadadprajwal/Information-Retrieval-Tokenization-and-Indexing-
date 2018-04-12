# Libraries and Packages used
from bs4 import BeautifulSoup
import os
import sys
import re
import time
import numpy as np
import matplotlib.pyplot as plt

# Taking the input folder name and output folder name from command line.
inputFolderName = sys.argv[1]
outputFolderName = sys.argv[2]

# Getting the path of the input folder.
# Setting the path of the output folder for storing the dictionary and posting files.
inputFolderPath = os.getcwd() + '/' + inputFolderName + '/'
outputFolderPath = os.getcwd() + '/' + outputFolderName + '/'

# Reading the stopwords file and storing into a list.
stopwords = [line.strip() for line in open('stopwords.txt')]

# Retrieving the total number of documents.
numberOfDocuments = len(os.listdir(inputFolderPath))

# Logic for checking if the input folder exists or not and creating the outputFolder.
if not os.path.exists(outputFolderPath):
    print('Output Folder Created.')
    os.makedirs(outputFolderPath)
else:
    print('Output Folder already exists.')

if not os.path.exists(inputFolderPath):
    print('Input folder files does not exists.')
else:
    print('Number of files =', numberOfDocuments)

# Dictionary to store the Number of documents with token 'T' in it.
documentCountDict = {}

# Set to store all the unique tokens.
termSet = set()

# A list of dictionaries where each dictionary stores the tokens and its frequency in it.
documentList = []

# Variable to store total time taken by all the files.
fileTimeElapsed1 = 0
fileTimeElapsed2 = 0
fileTimeElapsed3 = 0

print("Parsing the HTML files and generating tokens.")

# List for storing the name of the files.
filenameList = []

tokenGenerationTime = []
# Logic for calculating the tokens from a HTML file.
for filename in sorted(os.listdir(inputFolderPath)):
    modifiedFileName = os.path.splitext(filename)[0]

    # Storing the filename.
    filenameList.append(filename)

    # Noting the start time.
    start_time1 = time.time()

    # Opening the file.
    file = open(inputFolderPath + filename, 'r', errors='replace')

    # Using Beautiful Soup to parse HTML file.
    bs = BeautifulSoup(file, 'html.parser')

    # Extracting the text from the HTML file.
    text_from_html = bs.get_text()

    # Closing the file.
    file.close()

    # Removing the special characters.
    string = re.sub(r'[^\w]', ' ', text_from_html)

    # Removing the numbers.
    stringWithoutNumbers = re.sub('[0-9]', '', string)

    # Removing the underscore character.
    formattedString = re.sub('_', '', stringWithoutNumbers)

    # Splitting the sentence on space and generating tokens.
    spaceRemovedFileContent = formattedString.split()

    # Removing the stopwords.
    removedStopwords = [term for term in spaceRemovedFileContent if term not in stopwords]

    # Removing blank entries from the variable.
    tokens = [x for x in removedStopwords if x]

    # A dictionary to store the tokens and counting their frequencies.
    wordDict = {}
    for word in tokens:
        lowercaseWord = word.lower()
        if 2 < len(str(lowercaseWord)) < 10:
            if lowercaseWord not in stopwords:
                termSet.add(lowercaseWord)
                wordDict[lowercaseWord] = wordDict.get(lowercaseWord, 0) + 1

    # Noting the end time and calculating the time elapsed.
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    fileTimeElapsed1 = fileTimeElapsed1 + elapsed_time1
    tokenGenerationTime.append(fileTimeElapsed1)

    # Appending the dictionary into a list.
    documentList.append(wordDict)

# Dictionary to store the Inverted Index of the Terms.
invertedIndexDict = {}

print("Tokens generated.")
print("Number of unique terms = ", len(termSet))

# Logic for generating the term inverted index.
documentID = 1
invertedIndexTime = []
for term in termSet:
    listOfTuples = []
    for document in documentList:
        start_time2 = time.time()
        if term in document.keys():
            if term not in stopwords:
                termValueTuple = [documentList.index(document), document[term]]
                listOfTuples.append(termValueTuple)
        end_time2 = time.time()
        elapsed_time2 = end_time2 - start_time2
        fileTimeElapsed2 = fileTimeElapsed2 + elapsed_time2
        invertedIndexTime.append(fileTimeElapsed2)
    invertedIndexDict[term] = listOfTuples


# Calculating the weight of the terms(TFIDF values).
tfidfTime = []
for term in invertedIndexDict.keys():
    tupleList = invertedIndexDict[term]
    for termTuple in tupleList:
        start_time3 = time.time()
        wordFreq = termTuple[1]
        documentIndex = termTuple[0]
        totalWords = len(documentList[documentIndex])
        TF = wordFreq / totalWords
        IDF = numberOfDocuments / len(invertedIndexDict[term])
        TFIDF = "%.3f" % (TF * IDF)
        termTuple[1] = TFIDF
        end_time3 = time.time()
        elapsed_time3 = end_time3 - start_time3
        fileTimeElapsed3 = fileTimeElapsed3 + elapsed_time3
        tfidfTime.append(fileTimeElapsed3)

print("Inverted Index Generated")
print("Size of Inverted Index = ", len(invertedIndexDict.keys()), "terms")

# Creating the Dictionary File.
dictionaryFile = open(outputFolderPath + "DictionaryFile.txt", 'w')
termPositionCounter = 1
dictionaryFile.write("The word" + "\n" +
                     "The number of documents that contain that word (this corresponds to the number of "
                     "records that word gets in the postings file)" + "\n" +
                     "The location of the first record for that word in the postings file" + "\n\n\n")
for term in sorted(invertedIndexDict):
    dictionaryFile.write(term + "\n" + str(len(invertedIndexDict[term])) + "\n" + str(termPositionCounter) + "\n")
    termPositionCounter += len(invertedIndexDict[term])
dictionaryFile.close()

print("Dictionary file created.")

# Creating the Postings File.
postingFile = open(outputFolderPath + "PostingFile.txt", 'w')
postingFile.write("The document id" + ", " + "The normalized weight of the word in the document" + "\n\n\n")
for termList in sorted(invertedIndexDict):
    for docs in sorted(invertedIndexDict[termList]):
        postingFile.write(str(filenameList[docs[0]]) + ", " + str(docs[1]) + "\n")
postingFile.close()

print("Postings file created.")

# Plotting the Number of Files vs Total time taken graph.
x_axis = np.logspace(np.log10(1), np.log10(numberOfDocuments), dtype=int, num=len(os.listdir(inputFolderPath)))
y_axis = list(map(sum, zip(tokenGenerationTime, invertedIndexTime, tfidfTime)))

plt.plot(x_axis, y_axis)
plt.xlabel("Number of Files")
plt.ylabel("Total Time Elapsed (sec)")
plt.show()
