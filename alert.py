import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# ALERT CONFIGURATION

def get_bool_env(var, default=False):
    val = os.getenv(var)
    if val is None:
        return default
    return str(val).lower() in ['true', '1', 'yes']

EMAIL_ALERT = get_bool_env('EMAIL_ALERT', True)
PHONE_ALERT = get_bool_env('PHONE_ALERT', True)
GOV_ALERT = get_bool_env('GOV_ALERT', True)

EMAIL_TO = os.getenv('EMAIL_TO', 'recipient@example.com')
EMAIL_TO_LIST = [email.strip() for email in EMAIL_TO.split(',') if email.strip()]
EMAIL_FROM = os.getenv('EMAIL_FROM', 'your_email@example.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.example.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
PHONE_NUMBER = os.getenv('PHONE_NUMBER', '+911234567890')
GOV_CONTACT = os.getenv('GOV_CONTACT', 'gov-alert@example.com')

def send_alert(report):
    print("[DEBUG] send_alert called with report:", report)
    print("[DEBUG] EMAIL_ALERT:", EMAIL_ALERT)
    print("[DEBUG] EMAIL_TO_LIST:", EMAIL_TO_LIST)
    print("[DEBUG] EMAIL_FROM:", EMAIL_FROM)
    print("[DEBUG] EMAIL_PASSWORD (set):", bool(EMAIL_PASSWORD))
    print("[DEBUG] SMTP_SERVER:", SMTP_SERVER)
    print("[DEBUG] SMTP_PORT:", SMTP_PORT)
    """Send alert if water is unsafe. Accepts a report dict."""
    alert_message = f"""
ALERT: WATER SAFETY ISSUE DETECTED

Water Safety Status: {report.get('Water Safety', 'Unknown')}
Predicted Disease: {report.get('Predicted Disease', 'Unknown')}
Outbreak Probability: {report.get('Outbreak Probability', 'Unknown')}
Risk Factors: {report.get('Risk Factors', 'Unknown')}
Recommendation: {report.get('Recommendation', 'Unknown')}
Top Contributing Features: {report.get('Top Contributing Features', 'N/A')}
"""
    if str(report.get('Water Safety', '')).lower() == 'unsafe':
        print("\n[ALERT] Water is UNSAFE! Sending notifications...")
        # Email Alert
        if EMAIL_ALERT and EMAIL_TO_LIST and EMAIL_FROM and EMAIL_PASSWORD and SMTP_SERVER:
            print("[DEBUG] All email config present. Attempting to send email...")
            try:
                msg = MIMEMultipart()
                msg['From'] = EMAIL_FROM
                msg['To'] = ', '.join(EMAIL_TO_LIST)
                msg['Subject'] = 'URGENT: Unsafe Water Detected!'
                # Format alert message for email (plain text, clear labels)
                email_body = (
                    "==== SMART COMMUNITY HEALTH ALERT ===="
                    f"\n\nALERT: WATER SAFETY ISSUE DETECTED"
                    f"\n\nWater Safety Status      : {report.get('Water Safety', 'Unknown')}"
                    f"\nPredicted Disease        : {report.get('Predicted Disease', 'Unknown')}"
                    f"\nOutbreak Probability     : {report.get('Outbreak Probability', 'Unknown')}"
                    f"\nRisk Factors             : {report.get('Risk Factors', 'Unknown')}"
                    f"\nRecommendation           : {report.get('Recommendation', 'Unknown')}"
                    f"\nTop Contributing Features: {report.get('Top Contributing Features', 'N/A')}"
                    "\n========================================\n"
                )
                msg.attach(MIMEText(email_body, 'plain'))
                server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15)
                server.starttls()
                server.login(EMAIL_FROM, EMAIL_PASSWORD)
                server.sendmail(EMAIL_FROM, EMAIL_TO_LIST, msg.as_string())
                server.quit()
                print(f"[EMAIL] Alert sent to {', '.join(EMAIL_TO_LIST)}")
            except smtplib.SMTPAuthenticationError as e:
                print(f"[EMAIL] Authentication failed: {e}")
            except smtplib.SMTPConnectError as e:
                print(f"[EMAIL] Connection failed: {e}")
            except smtplib.SMTPException as e:
                print(f"[EMAIL] SMTP error: {e}")
            except Exception as e:
                print(f"[EMAIL] Unexpected error: {type(e).__name__}: {e}")
        else:
            print("[EMAIL] Email alert not sent: missing configuration or disabled.")
        # Phone Alert (placeholder: print to console)
        if PHONE_ALERT and PHONE_NUMBER:
            print(f"[PHONE] SMS to {PHONE_NUMBER}: {alert_message}")
        # Government Alert (placeholder: print to console)
        if GOV_ALERT and GOV_CONTACT:
            print(f"[GOV] Notification sent to government: {GOV_CONTACT}\n{alert_message}")
    else:
        print("\n[INFO] Water is safe. No alert sent.")

if __name__ == "__main__":
    # Example unsafe report for testing
    test_report = {
        'Water Safety': 'Unsafe',
        'Predicted Disease': 'Cholera',
        'Outbreak Probability': '80%',
        'Risk Factors': 'High turbidity',
        'Recommendation': 'Immediate chlorination',
        'Top Contributing Features': 'turbidity: 0.8, pH: 0.6'
    }
    print("\n=== TESTING ALERT SYSTEM ===")
    send_alert(test_report)
    print("\n=== ALERT TEST COMPLETE ===")
