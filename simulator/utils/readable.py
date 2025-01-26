import time
from datetime import timedelta

def wait_animation(duration: float) -> None:
    t0 = time.time()
    seq = ['   ','.  ','.. ','...']
    L = len(seq)
    k = 0
    elapsed = 0.0
    while elapsed < duration:
        elapsed = time.time() - t0
        print(f"Waiting {duration-elapsed:.1f} seconds", seq[k%L], end='\r')
        time.sleep(0.5)
        k += 1
    print()
    
    
def seconds_to_hhmmss(t: float) -> str:
    td = timedelta(seconds=t)
    if td.microseconds == 0.0:
        return td.__str__() + ".00"
    else:
        return td.__str__()[:-4]

def seconds_to_dhms(t: float) -> str:
    txt = ""
    
    days = t // 86400.0
    if days > 0:
        txt += f"{days:.0f} d "
    t -= days * 86400.0

    hours = t // 3600.0
    if hours > 0 or days > 0:
        txt += f"{hours:.0f} h "
    t -= hours * 3600.0
        
    mins = t // 60.0
    if mins > 0 or hours > 0 or days > 0:
        txt += f"{mins:.0f} min "
    t -= mins * 60.0

    txt += f"{t:.2f} s"
    
    return txt

    
if __name__ == "__main__":
    t = 0.0
    while True:
        t += 12.34
        print(seconds_to_dhms(t), seconds_to_hhmmss(t))