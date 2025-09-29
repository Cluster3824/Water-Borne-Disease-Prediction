import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ALERT CONFIGURATION
EMAIL_ALERT = True
PHONE_ALERT = True  # Placeholder: prints to console
GOV_ALERT = True   # Placeholder: prints to console

# Contact details (replace with real values)
EMAIL_TO = os.getenv('EMAIL_TO', 'recipient@example.com')
if ',' in EMAIL_TO:
    EMAIL_TO_LIST = [email.strip() for email in EMAIL_TO.split(',')]
else:
    EMAIL_TO_LIST = [EMAIL_TO]
EMAIL_FROM = os.getenv('EMAIL_FROM', 'your_email@example.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'your_email_password')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.example.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
PHONE_NUMBER = os.getenv('PHONE_NUMBER', '+911234567890')  # Placeholder
GOV_CONTACT = os.getenv('GOV_CONTACT', 'gov-alert@example.com')  # Placeholder

def send_alert(report):
    """Send alert if water is unsafe. Accepts a report dict."""
    alert_message = f"""
ALERT: WATER SAFETY ISSUE DETECTED

Water Safety Status: {report['Water Safety']}
Predicted Disease: {report['Predicted Disease']}
Outbreak Probability: {report['Outbreak Probability']}
Risk Factors: {report['Risk Factors']}
Recommendation: {report['Recommendation']}
"""
    if report['Water Safety'].lower() == 'unsafe':
        print("\n[ALERT] Water is UNSAFE! Sending notifications...")
        # Email Alert
        if EMAIL_ALERT:
            try:
                msg = MIMEMultipart()
                msg['From'] = EMAIL_FROM
                msg['To'] = ', '.join(EMAIL_TO_LIST)
                msg['Subject'] = 'URGENT: Unsafe Water Detected!'
                msg.attach(MIMEText(alert_message, 'plain'))
                server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.sendmail(EMAIL_FROM, EMAIL_TO_LIST, msg.as_string())
                server.quit()
                print(f"[EMAIL] Alert sent to {', '.join(EMAIL_TO_LIST)}")
            except Exception as e:
                print(f"[EMAIL] Failed to send: {e}")
        # Phone Alert (placeholder)
        if PHONE_ALERT:
            print(f"[PHONE] SMS to {PHONE_NUMBER}: {alert_message}")
        # Government Alert (placeholder)
        if GOV_ALERT:
            print(f"[GOV] Notification sent to government: {GOV_CONTACT}\n{alert_message}")
    else:
        print("\n[INFO] Water is safe. No alert sent.")
