from datetime import datetime

global logfile
global also_stdout
global log_messages

logfile = None
also_stdout = True
log_messages = []


def set_logfile(path: str):
    global logfile
    logfile = path


def log(component: str, message: str):
    global also_stdout
    global log_messages

    log_str = (f"[{datetime.now().strftime('%H:%M:%S')} | {component}] {message}")
    if also_stdout:
        print(log_str)
    log_messages.append(log_str)
    if len(log_messages) > 10:
        write_log()


def write_log():
    global logfile
    global log_messages
    if logfile is not None:
        with open(logfile, 'a') as file:
            for msg in log_messages:
                file.write(f"{msg}\n")
            log_messages = []
