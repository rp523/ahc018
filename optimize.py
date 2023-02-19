import subprocess
from tqdm import tqdm

def main():
    subprocess.getoutput("cargo build --release")
    subprocess.getoutput("cd tools && cargo build --release")

    score_sum = 0
    score_norm = 0
    score_worst = -1
    worst_case = 0
    for i in tqdm(range(3000)):
        cmd = "./tools/target/release/tester target/release/start < tools/in/{0:04d}.txt > tools/out/out{0:04d}.txt".format(i, i)
        ret = subprocess.getoutput(cmd)
        keywd = "Total Cost = "
        score = int(ret[ret.find(keywd) + len(keywd):])
        score_sum += score
        score_norm += 1
        if (score_worst < 0) or (score_worst < score):
            score_worst = score
            worst_case = i
    print("average", score_sum / score_norm)
    print("worst", worst_case, score_worst)
if __name__ == "__main__":
    main()