import os
import re
import json
import pandas as pd
import datetime

currentDate = datetime.datetime.now()

pathToData = 'data/'
pathToDataBeforeXLS = 'dataBeforeProcessingXLS/'
pathToDataAfterXLS = 'dataAfterProcessingXLS/'
pathToDataAfterCSV = 'dataAfterProcessingCSV/'

uselessColumns = ['id_str', 'utc_offset', 'time_zone', 'lang', 'contributors_enabled', 'is_translator',
                  'is_translation_enabled', 'profile_background_tile',
                  'entities.description.urls', 'entities.url.urls', 'translator_type', 'notifications',
                  'profile_image_url_https', 'has_extended_profile',
                  'profile_background_image_url_https']

pd.set_option("display.max_rows", None, "display.max_columns", None)

trueDataFileName = pathToData + 'verified-2019_tweets.json'
botsDataFileName = pathToData + 'botwiki-2019_tweets.json'
trueLabelsFileName = pathToData + 'verified-2019.tsv'
botsLabelsFileName = pathToData + 'botwiki-2019.tsv'

politicalBotsFileName = pathToData + 'political-bots-2019_tweets.json'
celebrityFileName = pathToData + 'celebrity-2019_tweets.json'
politicalBotsLabelsFileName = pathToData + 'political-bots-2019.tsv'
celebrityLabelsFileName = pathToData + 'celebrity-2019.tsv'


def getIntFromBoolean(boolValue):
    if boolValue:
        return 1
    else:
        return 0


def countUserAge(createdAt):
    dataFormat = '%a %b %d %H:%M:%S %z %Y'
    registrationDate = datetime.datetime.strptime(str(createdAt), dataFormat)
    return int((currentDate.date() - registrationDate.date()).days)


def getNumberOfDigitsFromString(string):
    return sum(c.isdigit() for c in string)


def isItBot(string):
    if string == 'bot':
        return 1
    else:
        return 0


def getFollowersFriendsRatio(followers, friends, maxNumberOfFriends):
    if friends != 0:
        return followers / friends
    else:
        return maxNumberOfFriends * 2


def createFinalDataFrom(dataBeforeProcessing, bot_or_human):
    finalData = pd.DataFrame()
    maxNumberOfFriends = dataBeforeProcessing['friends_count'].max()
    finalData['statuses_count'] = dataBeforeProcessing['statuses_count']
    finalData['followers_count'] = dataBeforeProcessing['followers_count']
    finalData['friends_count'] = dataBeforeProcessing['friends_count']
    finalData['favourites_count'] = dataBeforeProcessing['favourites_count']
    finalData['listed_count'] = dataBeforeProcessing['listed_count']
    finalData['is_default_profile'] = dataBeforeProcessing.apply(lambda x: getIntFromBoolean(x['default_profile']),
                                                                 axis=1)
    finalData['is_profile_use_background_image'] = dataBeforeProcessing.apply(
        lambda x: getIntFromBoolean(x['profile_use_background_image']), axis=1)
    finalData['is_verified'] = dataBeforeProcessing.apply(lambda x: getIntFromBoolean(x['verified']), axis=1)
    finalData['user_age'] = dataBeforeProcessing.apply(lambda x: countUserAge(x['created_at']), axis=1)
    finalData['tweets_freq'] = dataBeforeProcessing.apply(lambda x: x['statuses_count'] / countUserAge(x['created_at']),
                                                          axis=1)
    finalData['followers_growth_rate'] = dataBeforeProcessing.apply(
        lambda x: x['followers_count'] / countUserAge(x['created_at']), axis=1)
    finalData['friends_growth_rate'] = dataBeforeProcessing.apply(
        lambda x: x['friends_count'] / countUserAge(x['created_at']),
        axis=1)
    finalData['favourites_growth_rate'] = dataBeforeProcessing.apply(
        lambda x: x['favourites_count'] / countUserAge(x['created_at']), axis=1)
    finalData['listed_growth_rate'] = dataBeforeProcessing.apply(
        lambda x: x['listed_count'] / countUserAge(x['created_at']),
        axis=1)
    finalData['followers_friends_ratio'] = dataBeforeProcessing.apply(
        lambda x: getFollowersFriendsRatio(x['followers_count'], x['friends_count'], maxNumberOfFriends), axis=1)
    finalData['screen_name_length'] = dataBeforeProcessing.apply(lambda x: len(x['screen_name']), axis=1)
    finalData['num_digits_in_screen_name'] = dataBeforeProcessing.apply(
        lambda x: getNumberOfDigitsFromString(x['screen_name']),
        axis=1)
    finalData['length_of_name'] = dataBeforeProcessing.apply(lambda x: len(x['name']), axis=1)
    finalData['num_digits_in_name'] = dataBeforeProcessing.apply(lambda x: getNumberOfDigitsFromString(x['name']),
                                                                 axis=1)
    finalData['description_length'] = dataBeforeProcessing.apply(lambda x: len(x['description']), axis=1)
    finalData['is_bot'] = dataBeforeProcessing.apply(lambda x: isItBot(x[bot_or_human]), axis=1)
    return finalData


def getDataFromJsonAndTsv(dataFileName, labelsFilName, bot_or_human):
    data = pd.read_json(dataFileName)
    data = pd.json_normalize(data['user']).drop(uselessColumns, axis=1)
    data = data.drop_duplicates(subset=['id'])

    labels = pd.read_csv(labelsFilName, sep='\t')
    labels.columns = ['id', bot_or_human]

    return pd.merge(data, labels, how='inner', on='id')


def saveData(data, name, before_or_after):
    if before_or_after == 'before':
        data.to_excel(pathToDataBeforeXLS + name + '.xls')
    else:
        data.to_excel(pathToDataAfterXLS + name + '.xls')
        data.to_csv(pathToDataAfterCSV + name + '.csv')


beforeName = 'BeforeProcessing'
afterName = 'AfterProcessing'

bots = getDataFromJsonAndTsv(botsDataFileName, botsLabelsFileName, 'bot')
humans = getDataFromJsonAndTsv(trueDataFileName, trueLabelsFileName, 'human')
celebrities = getDataFromJsonAndTsv(celebrityFileName, celebrityLabelsFileName, 'human')
politicalBots = getDataFromJsonAndTsv(politicalBotsFileName, politicalBotsLabelsFileName, 'bot')
saveData(bots, 'bots' + beforeName, 'before')
saveData(humans, 'humans' + beforeName, 'before')
saveData(celebrities, 'celebrities' + beforeName, 'before')
saveData(politicalBots, 'politicalBots' + beforeName, 'before')

botsFinalData = createFinalDataFrom(bots, 'bot')
humansFinalData = createFinalDataFrom(humans, 'human')
celebritiesFinalData = createFinalDataFrom(celebrities, 'human')
politicalBotsFinalData = createFinalDataFrom(politicalBots, 'bot')
saveData(botsFinalData, 'bots' + afterName, 'after')
saveData(humansFinalData, 'humans' + afterName, 'after')
saveData(celebritiesFinalData, 'celebrities' + afterName, 'after')
saveData(politicalBotsFinalData, 'politicalBots' + afterName, 'after')



