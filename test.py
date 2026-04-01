import time

print("test.py __name__:", __name__)

def sleeper(I):
    for i in range(0,I):
        print(i)
        time.sleep(1)



if __name__=="__main__":
    sleeper(5)
