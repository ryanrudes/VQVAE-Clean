from rich import print
import smtplib
import re

regex = '^[a-z0-9](\.?[a-z0-9]){5,}@gmail\.com$'

def valid(address):
    return bool(re.search(regex, address))

def validate(address):
    if not valid(address):
        raise ValueError('Invalid mail address. Expected `address` to be a valid email address ending in \'@gmail.com\'')

def send(body, address, password, subject, port=587):
    message = f"""\
From: {address}
To: {address}
Subject: {subject}

{body}
"""

    try:
        server = smtplib.SMTP('smtp.gmail.com', port)
        server.ehlo()
        server.starttls()
        server.login(address, password)
        server.sendmail(address, address, message)
    except Exception as e:
        print (str(e))
        print ("An error occurred when attempting to send the email")
        print ("Please refer to https://stackabuse.com/how-to-send-emails-with-gmail-using-python/ in order to resolve the problem")
    finally:
        server.close()
