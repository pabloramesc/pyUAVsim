import time

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