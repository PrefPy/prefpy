import operator
import random

# ALLOCATION ALGORITHM FUNCTIONS HERE:
def allocation(pollAlgorithm, itemList, responseList):
    #make sure there is at least one response  
    if len(responseList) == 0:
        return

    if pollAlgorithm == 1:
        #SD early first
        return allocation_serial_dictatorship(itemList, responseList, early_first = 1)
    elif pollAlgorithm == 2:
        #SD late first
        return allocation_serial_dictatorship(itemList, list(reversed(responseList)), early_first = 0)
    elif pollAlgorithm == 3:
        return allocation_manual(itemList, responseList)

# iterate through each response and assign the highest ranking choice that is still available 
# List<String> items
# List<(String, Dict<String, int>)> responses is a list of tuples, where the first entry is the user and the second
#     entry is the dictionary. Each dictionary item maps to a rank (integer) 
# return Dict<String, String> allocationResults which maps users to items
def getAllocationResults(items, responses):
    allocationResults = {}
    for response in responses:
        # no more items left to allocate
        if len(items) == 0:
            return
        
        highestRank = len(items)
        myitem = items[0]

        # here we find the item remaining that this user ranked the highest
        username = response[0]
        preferences = response[1]
        for item in items:
            if preferences.get(item) < highestRank:
                highestRank = preferences.get(item)
                myitem = item

        print ("Allocating item " + myitem + " to user " + username)
        # assign item
        allocationResults[username] = myitem
        # remove the item from consideration for other students
        items.remove(myitem)                
    return allocationResults

# Serial dictatorship algorithm to allocate items to students for a given question.
# It takes as an argument the response set to run the algorithm on.
# The order of the serial dictatorship will be decided by increasing
# order of the timestamps on the responses for novel questions, and reverse
# order of the timestamps on the original question for follow-up questions.
def allocation_serial_dictatorship(itemList, responseList, early_first = 1):
    return getAllocationResults(itemList, responseList)

# the poll owner can specify an order to allocate choices to voters
def allocation_manual(itemList, responseList):
    return getAllocationResults(itemList, responseList)