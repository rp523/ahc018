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
    delta_range: int,
    delta_cost_w: int,
    atk_eval_rate: int,
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
        cmd += " {}".format(delta_range)
        cmd += " {}".format(delta_cost_w)
        cmd += " {}".format(atk_eval_rate)
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
            "    Param {{eff: {}, key_power: {}, key_exca_th: {}, observe_power: {}, observe_exca_th: {}, connect_power: {}, connect_exca_th: {}, evalw: {}, fix_rate: {}, delta_range: {}, delta_cost_w: {}, atk_eval_rate: {}, }},,{},{},{}\n".format(
                eff,
                key_power,
                key_exca_th,
                observe_power,
                observe_exca_th,
                connect_power,
                connect_exca_th,
                evalw,
                fix_rate,
                delta_range,
                delta_cost_w,
                atk_eval_rate,
                ave_score,
                worst_case,
                score_worst,
            )
        )
    return ave_score

def objective0(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(0, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective1(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(1, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective2(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(2, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective3(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(3, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective4(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(4, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective5(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(5, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective6(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(6, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)
def objective7(trial):
    eff = trial.suggest_int("eff", 12, 18)
    key_power = trial.suggest_int("key_power", 1, 300, 3)
    key_exca_th = trial.suggest_int("key_exca_th", 1, 2500, 50)
    observe_power = trial.suggest_int("observe_power", 10, 150, 5)
    observe_exca_th = trial.suggest_int("observe_exca_th", 1, 150, 5)
    connect_power = trial.suggest_int("connect_power", 1, 150, 3)
    connect_exca_th = trial.suggest_int("connect_exca_th", 10, 400, 10)
    evalw = trial.suggest_int("evalw", 4, 36, 4)
    fix_rate = trial.suggest_int("fix_rate", 1, 256, 5)
    delta_range = trial.suggest_int("delta_range", 1, 2)
    delta_cost_w = trial.suggest_int("delta_cost_w", 1, 32, 3)
    atk_eval_rate = trial.suggest_int("atk_eval_rate", 1, 8)
    return calc_score(7, eff, key_power, key_exca_th, observe_power, observe_exca_th, connect_power, connect_exca_th, evalw, fix_rate, delta_range, delta_cost_w, atk_eval_rate)

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