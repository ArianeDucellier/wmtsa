"""
This module contains function to convert dates in appropriate format
"""

from datetime import datetime, timedelta

def matlab2ymdhms(time, roundsec=True):
    """
    Convert Matlab format to year/month/day/hour/minute/second

    Input:
        type time =  float
        time = Number of days since January 1, 0000 (Matlab format)
    Output:
        type output = tuple of 6 integers
        output = year, month, day, hour, minute, second, (microsecond)
    """    
    myday = datetime.fromordinal(int(time)) + \
        timedelta(days=time % 1) - timedelta(days=366)
    year = myday.year
    month = myday.month
    day = myday.day
    hour = myday.hour
    minute = myday.minute
    second = myday.second
    microsecond = myday.microsecond
    if (roundsec==True):
        rsecond = int(round(second + microsecond / 1000000.0))
        if (rsecond == 60):
            minute = minute + 1
            rsecond = 0
        if (minute == 60):
            hour = hour + 1
            minute = 0
        if (hour == 24):
            day = day + 1
            hour = 0
        if ((month in [1, 3, 5, 7, 8, 10, 12]) and (day == 32)):
            month = month + 1
            day = 1
        elif ((month in [4, 6, 9, 11]) and (day == 31)):
            month = month + 1
            day = 1
        else:
            if (((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0)):
                if (day == 30):
                    month = month + 1
                    day = 1
            else:
                if (day == 29):
                    month = month + 1
                    day = 1
        if (month == 13):
            year = year + 1
            month = 1
        return (year, month, day, hour, minute, rsecond)
    else:
        return (year, month, day, hour, minute, second, microsecond)

def ymdhms2matlab(year, month, day, hour, minute, second):
    """
    Convert year/month/day/hour/minute/second to Matlab format

    Input:
        type intput = 6 integers
        input = year, month, day, hour, minute, second
    Output:
        type time = float
        time = Number of days since January 1, 0000 (Matlab format)
    """
    myday = datetime(year, month, day, hour, minute, second)
    myday = myday + timedelta(days=366)
    frac = (myday - datetime(year, month, day, 0, 0, 0)).seconds / 86400.0
    myday = myday.toordinal() + frac
    return(myday) 

def string2day(day, hour):
    """
    Convert strings to number of years since 0 AD

    Input:
        type day = string
        day = 'YYYY-MM-DD'
        type hour = string
        hour = 'HH:mm:ss'
    Output:
        type date = float
        date = Numbers of years since 0 AD
    """
    YY = int(day[0 : 4])
    MM = int(day[5 : 7])
    DD = int(day[8 : 10])
    HH = int(hour[0 : 2])
    mm = int(hour[3 : 5])
    ss = int(hour[6 : 8])
    ndays = 0.0
    for i in range(1, MM):
        if (i in [1, 3, 5, 7, 8, 10, 12]):
            ndays = ndays + 31.0
        elif (i in [4, 6, 9, 11]):
            ndays = ndays + 30.0
        else:
            if (YY % 4 == 0):
                ndays = ndays + 29.0
            else:
                ndays = ndays + 28.0
    ndays = ndays + DD - 1
    fracday = ((HH - 1) * 3600.0 + (mm - 1) * 60.0 + ss) / 86400.0
    ndays = ndays + fracday
    if (YY % 4 == 0):
        fracyear = ndays / 366.0
    else:
        fracyear = ndays / 365.0
    date = YY + fracyear
    return date
