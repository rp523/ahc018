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
    key_power: int,
    key_exca_th: int,
    observe_power: int,
    observe_exca_th: int,
    connect_power: int,
    connect_exca_th: int,
    evalw: int,
    fix_rate: int,
    delta_range_inv: int,
    delta_cost_w: int,
):
    c = 1 << cb
#    subprocess.getoutput("cargo build --release")
#    subprocess.getoutput("cd tools && cargo build --release")

    score_sum = 0
    score_norm = 0
    score_worst = -1
    worst_case = 0
    for i in range(1000):
        cmd = "./tools/target/release/tester target/release/start"
        cmd += " {}".format(eff)
        cmd += " {}".format(key_power)
        cmd += " {}".format(key_exca_th)
        cmd += " {}".format(observe_power)
        cmd += " {}".format(observe_exca_th)
        cmd += " {}".format(connect_power)
        cmd += " {}".format(connect_exca_th)
        cmd += " {}".format(evalw)
        cmd += " {}".format(fix_rate)
        cmd += " {}".format(delta_range_inv)
        cmd += " {}".format(delta_cost_w)
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
            "    Param {{eff: {}, key_power: {}, key_exca_th: {}, observe_power: {}, observe_exca_th: {}, connect_power: {}, connect_exca_th: {}, evalw: {}, fix_rate: {}, delta_range_inv: {}, delta_cost_w: {}, }},,{},{},{}\n".format(
                eff,
                key_power,
                key_exca_th,
                observe_power,
                observe_exca_th,
                connect_power,
                connect_exca_th,
                evalw,
                fix_rate,
                delta_range_inv,
                delta_cost_w,
                ave_score,
                worst_case,
                score_worst,
            )
        )
    return ave_score

def objective0(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(0, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective1(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(1, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective2(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(2, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective3(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(3, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective4(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(4, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective5(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(5, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective6(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(6, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)
def objective7(trial):
    eff = trial.suggest_int("eff", 2, 30)
    key_power = trial.suggest_int("key_power", 1, 1000)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500)
    observe_power = trial.suggest_int("observe_power", 1, 1000)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 2500)
    connect_power = trial.suggest_int("connect_power", 1, 1000)
    connect_exca_th = trial.suggest_int("connect_exca_th", 1, 2500)
    evalw = trial.suggest_int("evalw", 1, 64)
    fix_rate = trial.suggest_int("fix_rate", 1, 256)
    delta_range_inv = trial.suggest_int("delta_range_inv", 2, 30)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32)
    return calc_score(7, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range_inv, delta_cost_w)

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
    #optimize(0)
    #return
    futures = []
    with ThreadPoolExecutor(max_workers = 8, thread_name_prefix="thread") as pool:
        for cb in range(8):
            future = pool.submit(optimize, cb)
            futures.append(future)
    for future in futures:
        print(future.result())

if __name__ == "__main__":
    main()