import datetime as dt
import pytz

class DatetimesUtils:

    @classmethod
    def date_from_str_to_datetime(date):
        pattern = '%Y-%m-%d %H:%M:%S'
        return dt.datetime.strptime(date, pattern)

    @classmethod
    def convert_tz(cls, datetime, from_tz, to_tz):
        datetime = datetime.replace(tzinfo=from_tz)
        datetime = datetime.astimezone(pytz.timezone(to_tz))
        return  datetime

    @classmethod
    def convert_tz(cls, datetime, from_tz, to_tz):

        datetime = datetime.replace(tzinfo=from_tz)
        datetime = datetime.astimezone(pytz.timezone(to_tz)).to_pydatetime()
        year = datetime.year
        month = datetime.month
        day = datetime.day
        hour = datetime.hour
        minute = datetime.minute
        second = datetime.second
        # datetime = str(year)+"-"+str(month)+"-"+str(day)+" "+str(hour)+":"+str(minute)+":"+str(second)
        datetime = str(dt.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second))
        # datetime = datetime.srtftime("%Y-%m-%d %H:%M:%S")
        # .srtftime("%Y-%m-%d %H:%M:%S")
        return datetime
