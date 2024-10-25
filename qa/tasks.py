from django.core.mail import send_mail
from django.conf import settings
from datetime import date,datetime
from django.contrib.auth.models import User
from .factcheck import fact_checker


def send_scheduled_mail():
    subject = 'NEWS from Fact Checker '+'{}'.format(date.today())
    m=fact_checker()
    message = """,

We hope this email finds you well.

Our system has detected a potentially misleading or false claim circulating online. Below are the details of the claim and its verification:

Claim: {}

URL: {}

Rating: {}

Claim Date: {}

If you have any questions or need assistance, feel free to reach out to our support team at fakenewssupport@gmail.com.

Best regards,
The Fake News Detection Team

""".format(m[0],m[3],m[1],m[2])
    # users = list(User.objects.values_list('username', flat=True))

    # recipient_list = list(User.objects.values_list('email', flat=True))
    user_email_tuples = list(User.objects.order_by('id').values_list('username', 'email'))

    try:
        for username, email in user_email_tuples:
            send_mail(
                subject,
                "Dear {}".format(username)+message,
                settings.EMAIL_HOST_USER,
                [email],
                fail_silently=False
            )
        print("Mail Sent to all Users successfully!!")
    except:
        print("Failed to Send Mail")