import schedule
import time
from datetime import datetime
from paper_trader import run_trader

print("🎯 Bullseye - Auto Scheduler Started")
print("=" * 40)
print("⏰ Scheduled runs:")
print("   → 9:15 AM IST (Market Open)")
print("   → 9:00 PM IST (Crypto Evening Check)")
print("   → Press Ctrl+C to stop")
print("=" * 40)

def run_with_log():
    print(f"\n⏰ Auto run triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 40)
    try:
        run_trader()
    except Exception as e:
        print(f"❌ Error during run: {e}")

# --- SCHEDULE RUNS ---
schedule.every(1).hours.do(run_with_log)  # checks every hour!!

# --- KEEP RUNNING ---
print(f"\n✅ Scheduler is live!! Waiting for next run...")
print(f"🕐 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Show next run time
next_run = schedule.next_run()
print(f"⏭️  Next run at : {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

while True:
    schedule.run_pending()
    time.sleep(30)  # Check every 30 seconds