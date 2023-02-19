import optuna
import subprocess
from tqdm import tqdm

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    score = (x - 2) ** 2
    print(x, score)
    return score

def main():
    worst_score = -1
    sum_score = 0
    nrm = 0
    worst_case = 0
    subprocess.getoutput("cargo build --release")
    subprocess.getoutput("cd tools && cargo build --release")
    for i in tqdm(range(0, 3000)):
        cmd = "./tools/target/release/tester target/release/start < tools/in/{0:04d}.txt".format(i)
        cmd += " > tools/out/out{0:04d}.txt".format(i)
        rets = subprocess.getoutput(cmd)
        for ret in rets.split("\n"):
            if ret.find("Total Cost = ") == 0:
                score = int(ret[len("Total Cost = "):])
                sum_score += score
                nrm += 1
                if (worst_score < 0) or (worst_score < score):
                    worst_score = score
                    worst_case = i
    print("worst", worst_case, worst_score)
    print("average", sum_score / nrm)
    return
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_params)

if __name__ == "__main__":
    main()
