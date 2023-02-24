from pathlib import Path

def main():
    tgt = Path("/home/isgsktyktt/work/rust")
    for scan_dir in tgt.glob("*"):
        sm_score = 0
        scores = []
        min_vals = [None] * 8
        for cb in range(8):
            c = 1 << cb
            csv_path = scan_dir.joinpath("optuna{}.csv".format(c))
            min_score = -1
            if csv_path.exists():
                for line in open(csv_path):
                    vals = line.split(",")
                    ave_score = float(vals[-3])
                    if (min_score < 0) or (min_score > ave_score):
                        min_score = ave_score
                        min_vals[cb] = vals.copy()
                sm_score += min_score
                scores.append(min_score)
        if len(scores) >= 1:
            print(scan_dir.name, sm_score, sm_score / len(scores))
            print(scores)
            for min_val in min_vals:
                print(min_val)
if __name__ == "__main__":
    main()