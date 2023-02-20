import subprocess
from tqdm import tqdm
import optuna

def objective(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(eff, power, exca_th, evalw, fix_rate, delta_range_inv)

def calc_score(
    eff: int,
    power: int,
    exca_th: int,
    evalw: int,
    fix_rate: int,
    delta_range_inv: int,
):
    subprocess.getoutput("cargo build --release")
    subprocess.getoutput("cd tools && cargo build --release")

    score_sum = 0
    score_norm = 0
    score_worst = -1
    worst_case = 0
    for _cnt in (range(0, 8 * 1000)):
        _cp = _cnt % 8
        i = _cnt // 8
        c = 1 << _cp
        cmd = "./tools/target/release/tester target/release/start"
        cmd += " {}".format(eff)
        cmd += " {}".format(power)
        cmd += " {}".format(exca_th)
        cmd += " {}".format(evalw)
        cmd += " {}".format(fix_rate)
        cmd += " {}".format(delta_range_inv)
        cmd += " < tools/in{0}/{0:04d}.txt".format(c, i)
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
            "{} {} {} {} {} {},,{},{},{}\n".format(
                eff,
                power,
                exca_th,
                evalw,
                fix_rate,
                delta_range_inv,
                ave_score,
                worst_case,
                score_worst,
            )
        )
    return ave_score

def main():
    #for eff in range(10, 20 + 1):
    #    print(calc_score(eff, 100, 100, 8, 128))
    #print(calc_score(14, 68, 43, 9, 241, 18)); return
    study = optuna.create_study()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(objective, n_trials=9999999999)

if __name__ == "__main__":
    main()