from joblib import Parallel, delayed
import time

a = 100

def pr(t, b):
	global a
	time.sleep(0.1)
	print([t, a])
	return t


if __name__ == '__main__':
	tt = Parallel(5)(delayed(pr)(t, t) for t in range(10))
	print(type(tt))
	print(tt)