"""Keep-awake request and Windows sleep/resume lookup (CONSTRAINTS.md Amendments 2026-11-20 (j)). No system setting is changed."""
import ctypes
import datetime
import subprocess

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
SLEEP_IDS = (1, 42, 107, 506, 507)  # Power-Troubleshooter 1 (resume); Kernel-Power 42 sleep, 107 resume, 506/507 Modern Standby


def keep_awake():
    """Blocks idle sleep while the calling thread lives (does not prevent a lid-close/forced sleep)."""
    try:
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except Exception:
        return False


def sleep_events(start, end):
    """[(iso time, event id)] of sleep/resume events in the System log between start and end (datetimes)."""
    fmt = "%Y-%m-%d %H:%M:%S"
    ps = (f"$s=[datetime]'{start:{fmt}}'; $e=[datetime]'{end:{fmt}}'; "
          "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power',"
          "'Microsoft-Windows-Power-Troubleshooter'; StartTime=$s; EndTime=$e} -ErrorAction SilentlyContinue | "
          f"Where-Object {{ $_.Id -in {','.join(map(str, SLEEP_IDS))} }} | "
          "ForEach-Object { $_.TimeCreated.ToString('s') + '|' + $_.Id }")
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                         capture_output=True, text=True, timeout=120).stdout
    return [tuple(l.strip().split("|")) for l in out.splitlines() if "|" in l]


if __name__ == "__main__":
    print(keep_awake(), sleep_events(datetime.datetime(2026, 11, 15, 22, 50), datetime.datetime(2026, 11, 15, 23, 10)))
