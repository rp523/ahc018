import subprocess
from tqdm import tqdm
import optuna

def objective(trial):
    eff = trial.suggest_int("eff", 2, 50)
    power = trial.suggest_int("power", 1, 5000)
    exca_th = trial.suggest_int("exca_th", 1, 5000)
    evalw = trial.suggest_int("eff", 1, 5000)
    return calc_score(eff, power, exca_th, evalw)

def calc_score(
    eff: int,
    power: int,
    exca_th: int,
    evalw: int,
):
    subprocess.getoutput("cargo build --release")
    subprocess.getoutput("cd tools && cargo build --release")

    score_sum = 0
    score_norm = 0
    score_worst = -1
    worst_case = 0
    for i in tqdm(range(30)):
        cmd = "./tools/target/release/tester target/release/start {} {} {} {}".format(eff, power, exca_th, evalw)
        cmd += " < tools/in/{0:04d}.txt".format(i)
        #cmd += " > tools/out/out{0:04d}.txt".format(i)
        ret = subprocess.getoutput(cmd)
        keywd = "Total Cost = "
        score = int(ret[ret.find(keywd) + len(keywd):])
        score_sum += score
        score_norm += 1
        if (score_worst < 0) or (score_worst < score):
            score_worst = score
            worst_case = i
    ave_score = score_sum / score_norm
    with open("optuna.csv", "a") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                eff,
                power,
                exca_th,
                evalw,
                ave_score,
                worst_case,
                score_worst,
            )
        )
    return ave_score

def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials=9999999999)

if __name__ == "__main__":
    main()