# myapp/scheduler.py
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from .tasks import send_scheduled_mail

def start_scheduler():
    scheduler = BackgroundScheduler()
    trigger = CronTrigger(hour=0, minute=44, second=40) 
    scheduler.add_job(send_scheduled_mail, trigger)
    scheduler.start()
