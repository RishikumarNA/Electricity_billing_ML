from flask import Flask, render_template, request
from twilio.rest import Client
import requests
from bs4 import BeautifulSoup
import smtplib
from email.message import EmailMessage
import ssl
import csv
from test import values

app = Flask(__name__)


@app.route('/')
def my_page():
    return render_template('final.html')

@app.route('/submit_form', methods=['POST'])
def my_page2():
    INV_NO = request.form['INV_NO']
    DAY1 = request.form['DAY1']
    MONTH1 = request.form['MONTH1']  
    YEAR1 = request.form['YEAR1']
    HOUR1 = request.form['HOUR1']
    MIN1 = request.form['MIN1']  
    
    values(INV_NO,DAY1,MONTH1,YEAR1,HOUR1,MIN1)
    
    write_to_csv(INV_NO,DAY1,MONTH1,YEAR1,HOUR1,MIN1)
    # Process the form data as needed

    # Call the send_sms() and email_sender() functions

    return render_template('submit.html')

def write_to_csv(INV_NO,DAY1,MONTH1,YEAR1,HOUR1,MIN1):
    with open('database.csv', mode='a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([INV_NO,DAY1,MONTH1,YEAR1,HOUR1,MIN1])


if __name__ == "__main__":
    app.run()
