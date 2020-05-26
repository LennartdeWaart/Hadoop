#!/usr/bin/python


@outputSchema("word:chararray")
def getCountry(index):
    if index == 'T':
        return 'Turkey'
    elif index == 'R':
        return 'Russia'
    elif index == 'G':
        return 'Greece'
    elif index == 'F':
        return 'France'
    elif index == 'E':
        return 'England'
    elif index == 'A':
        return 'America'
    elif index == 'I':
        return 'Italy'
