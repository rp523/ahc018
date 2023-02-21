import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
from tqdm import tqdm
import optuna

def calc_score(
    cb: int,
    eff: int,
    power: int,
    exca_th: int,
    evalw: int,
    fix_rate: int,
    delta_range_inv: int,
):
    c = 1 << cb
    subprocess.getoutput("cargo build --release")
    subprocess.getoutput("cd tools && cargo build --release")

    score_sum = 0
    score_norm = 0
    score_worst = -1
    worst_case = 0
    for i in range(1000):
        cmd = "./tools/target/release/tester target/release/start"
        cmd += " {}".format(eff)
        cmd += " {}".format(power)
        cmd += " {}".format(exca_th)
        cmd += " {}".format(evalw)
        cmd += " {}".format(fix_rate)
        cmd += " {}".format(delta_range_inv)
        cmd += " < tools/in{}".format(c) + "/{0:04d}.txt".format(i)
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
    with open("optuna{}.csv".format(c), "a") as f:
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

def objective0(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(0, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective1(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(1, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective2(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(2, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective3(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(3, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective4(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(4, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective5(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(5, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective6(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(6, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
def objective7(trial):
    eff = trial.suggest_int("eff", 2, 30)
    power = trial.suggest_int("power", 1, 1000)
    exca_th = trial.suggest_int("exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    return calc_score(7, eff, power, exca_th, evalw, fix_rate, delta_range_inv)
objectives = [
    objective0,
    objective1,
    objective2,
    objective3,
    objective4,
    objective5,
    objective6,
    objective7,
]

def optimize(cb):
    #for eff in range(10, 20 + 1):
    #    print(calc_score(eff, 100, 100, 8, 128))
    #print(calc_score(14, 68, 43, 9, 241, 18)); return
    study = optuna.create_study()
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(objectives[cb], n_trials=9999999999)

def main():
    futures = []
    with ThreadPoolExecutor(max_workers = 4, thread_name_prefix="thread") as pool:
        for cb in range(8):
            future = pool.submit(optimize, cb)
            futures.append(future)
    for future in futures:
        print(future.result())

if __name__ == "__main__":
    main()