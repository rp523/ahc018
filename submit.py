from pathlib import Path
import subprocess
import time

class Submitter:
    def __init__(self) -> None:
        self.min_sm_score = 99999999999999999999
    def submit(self):
        tgt = Path("/home/isgsktyktt/work/rust/ahc018_scan10")
        sm_score = 0
        scores = []
        min_lines = [None] * 8
        for cb in range(8):
            c = 1 << cb
            csv_path = tgt.joinpath("optuna{}.csv".format(c))
            min_score = -1
            if csv_path.exists():
                for line in open(csv_path):
                    vals = line.split(",")
                    ave_score = float(vals[-3])
                    if (min_score < 0) or (min_score > ave_score):
                        min_score = ave_score
                        min_lines[cb] = line
                sm_score += min_score
                scores.append(min_score)
        if len(scores) >= 8:
            if self.min_sm_score > sm_score:
                self.min_sm_score = sm_score
                ##
                with open("submit.rs", "w") as fw:
                    for (i, line) in enumerate(open(tgt.joinpath("src", "main.rs"))):
                        if i == 263 - 1:
                            fw.write("// {} {}\n".format(self.min_sm_score, self.min_sm_score / 8))
                        elif (265 - 1 <= i) and (i <= 272 - 1):
                            min_line = min_lines[i - (265 - 1)]
                            fw.write(min_line[:min_line.rfind("},") + len("},")])
                            fw.write("\n")
                        else:
                            fw.write(line)
                subprocess.getoutput("git add submit.rs")
                subprocess.getoutput("git commit -m {}_{}".format(self.min_sm_score, self.min_sm_score / 8))
                subprocess.getoutput("git push")                
                ##

def main():
    submitter = Submitter()
    while True:
        try:
            submitter.submit()
        except:
            pass
        time.sleep(15 * 60)

if __name__ == "__main__":
    main()