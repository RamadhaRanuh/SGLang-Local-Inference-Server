import psutil
import os
import signal

def find_process_by_port(port):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

# The port number your SGLang server is using
port = 35526  # replace with your actual port number

process = find_process_by_port(port)
if process:
    print(f"Found server process: PID={process.pid}")
    try:
        process.terminate()
        process.wait(timeout=5)
        print(f"Server with PID {process.pid} terminated successfully")
    except:
        print(f"Forcefully killing process {process.pid}")
        if os.name == 'nt':  # Windows
            os.system(f"taskkill /PID {process.pid} /F")
        else:  # Unix/Linux/MacOS
            os.kill(process.pid, signal.SIGKILL)
else:
    print(f"No process found listening on port {port}")
