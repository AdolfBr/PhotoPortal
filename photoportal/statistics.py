"""Local counters and best-effort network synchronization."""
from datetime import datetime, timedelta
import logging, os, random, time
from .runtime import APPDATA_LOCAL

LOCAL_STATS_PATH = os.path.join(APPDATA_LOCAL, "PhotoPortal_daily_stats.txt")
NETWORK_STATS_PATH = r"\\zeu.local\SYSVOL\zeu.local\scripts\Software\scripts\checks\Photo_CKM\photo_stats.csv"

def update_local_daily_stats(path=LOCAL_STATS_PATH):
    today = time.strftime("%Y-%m-%d"); stats = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip(): date, count = line.strip().split(","); stats[date] = int(count)
    stats[today] = stats.get(today, 0) + 1
    with open(path, "w", encoding="utf-8") as stream:
        for date, count in stats.items(): stream.write(f"{date},{count}\n")

def sync_network_stats(local_path=LOCAL_STATS_PATH, network_path=NETWORK_STATS_PATH):
    if not os.path.exists(local_path): return
    hostname = os.environ.get("COMPUTERNAME", "UnknownPC"); cutoff = datetime.now() - timedelta(days=30)
    try:
        local = {}
        with open(local_path, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    date, count = line.strip().split(",")
                    if datetime.strptime(date, "%Y-%m-%d") >= cutoff: local[date] = int(count)
        network = {}
        if os.path.exists(network_path):
            with open(network_path, encoding="utf-8") as stream:
                for line in list(stream)[1:]:
                    host, date, count = line.strip().split(",")
                    if datetime.strptime(date, "%Y-%m-%d") >= cutoff: network[(host, date)] = int(count)
        for date, count in local.items(): network[(hostname, date)] = max(count, network.get((hostname, date), 0)); local[date] = network[(hostname, date)]
        time.sleep(random.uniform(.1, .5))
        os.makedirs(os.path.dirname(network_path), exist_ok=True)
        with open(network_path, "w", encoding="utf-8") as stream:
            stream.write("Hostname,Date,PhotoCount\n")
            for (host, date), count in network.items(): stream.write(f"{host},{date},{count}\n")
        with open(local_path, "w", encoding="utf-8") as stream:
            for date, count in local.items(): stream.write(f"{date},{count}\n")
    except Exception: logging.exception("Ошибка при синхронизации статистики")
