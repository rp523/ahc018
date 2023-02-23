#![allow(unused_macros, unused_imports, dead_code)]
use std::any::TypeId;
use std::cmp::{max, min, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::mem::swap;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign};

macro_rules! __debug_impl {
    ($x:expr) => {
        eprint!("{}={}  ", stringify!($x), &$x);
    };
    ($x:expr, $($y:expr),+) => (
        __debug_impl!($x);
        __debug_impl!($($y),+);
    );
}
macro_rules! __debug_line {
    () => {
        eprint!("L{}  ", line!());
    };
}
macro_rules! __debug_select {
    () => {
        eprintln!();
    };
    ($x:expr) => {
        __debug_line!();
        __debug_impl!($x);
        eprintln!();
    };
    ($x:expr, $($y:expr),+) => (
        __debug_line!();
        __debug_impl!($x);
        __debug_impl!($($y),+);
        eprintln!();
    );
}
macro_rules! debug {
    () => {
        if cfg!(debug_assertions) {
            __debug_select!();
        }
    };
    ($($xs:expr),+) => {
        if cfg!(debug_assertions) {
            __debug_select!($($xs),+);
        }
    };
}

mod change_min_max {
    pub trait ChangeMinMax<T> {
        fn chmin(&mut self, rhs: T) -> bool;
        fn chmax(&mut self, rhs: T) -> bool;
    }
    impl<T: PartialOrd + Copy> ChangeMinMax<T> for T {
        fn chmin(&mut self, rhs: T) -> bool {
            if *self > rhs {
                *self = rhs;
                true
            } else {
                false
            }
        }
        fn chmax(&mut self, rhs: T) -> bool {
            if *self < rhs {
                *self = rhs;
                true
            } else {
                false
            }
        }
    }
    impl<T: PartialOrd + Copy> ChangeMinMax<T> for Option<T> {
        fn chmin(&mut self, rhs: T) -> bool {
            if let Some(val) = *self {
                if val > rhs {
                    *self = Some(rhs);
                    true
                } else {
                    false
                }
            } else {
                *self = Some(rhs);
                true
            }
        }
        fn chmax(&mut self, rhs: T) -> bool {
            if let Some(val) = *self {
                if val < rhs {
                    *self = Some(rhs);
                    true
                } else {
                    false
                }
            } else {
                *self = Some(rhs);
                true
            }
        }
    }
}
use change_min_max::ChangeMinMax;

mod union_find {
    #[derive(Debug, Clone)]
    pub struct UnionFind {
        pub graph: Vec<Vec<usize>>,
        parents: Vec<usize>,
        grp_sz: Vec<usize>,
        grp_num: usize,
    }

    impl UnionFind {
        pub fn new(sz: usize) -> Self {
            Self {
                graph: vec![vec![]; sz],
                parents: (0..sz).collect::<Vec<usize>>(),
                grp_sz: vec![1; sz],
                grp_num: sz,
            }
        }
        pub fn root(&mut self, v: usize) -> usize {
            if v == self.parents[v] {
                v
            } else {
                self.parents[v] = self.root(self.parents[v]);
                self.parents[v]
            }
        }
        pub fn same(&mut self, a: usize, b: usize) -> bool {
            self.root(a) == self.root(b)
        }
        pub fn unite(&mut self, into: usize, from: usize) {
            self.graph[into].push(from);
            self.graph[from].push(into);
            let r_into = self.root(into);
            let r_from = self.root(from);
            if r_into != r_from {
                self.parents[r_from] = r_into;
                self.grp_sz[r_into] += self.grp_sz[r_from];
                self.grp_sz[r_from] = 0;
                self.grp_num -= 1;
            }
        }
        pub fn group_num(&self) -> usize {
            self.grp_num
        }
        pub fn group_size(&mut self, a: usize) -> usize {
            let ra = self.root(a);
            self.grp_sz[ra]
        }
    }
}
use union_find::UnionFind;

mod usize_move_delta {
    pub trait MoveDelta<T> {
        fn move_delta(self, delta: T, lim_lo: usize, lim_hi: usize) -> Option<usize>;
    }
    impl<T: Copy + Into<i64>> MoveDelta<T> for usize {
        fn move_delta(self, delta: T, lim_lo: usize, lim_hi: usize) -> Option<usize> {
            let delta: i64 = delta.into();
            let added: i64 = self as i64 + delta;
            let lim_lo: i64 = lim_lo as i64;
            let lim_hi: i64 = lim_hi as i64;
            if (lim_lo <= added) && (added <= lim_hi) {
                Some(added as usize)
            } else {
                None
            }
        }
    }
}
use usize_move_delta::MoveDelta;

mod procon_reader {
    use std::fmt::Debug;
    use std::io::Read;
    use std::str::FromStr;
    pub fn read<T: FromStr>() -> T
    where
        <T as FromStr>::Err: Debug,
    {
        let stdin = std::io::stdin();
        let mut stdin_lock = stdin.lock();
        let mut u8b: [u8; 1] = [0];
        loop {
            let mut buf: Vec<u8> = Vec::with_capacity(16);
            loop {
                let res = stdin_lock.read(&mut u8b);
                if res.unwrap_or(0) == 0 || u8b[0] <= b' ' {
                    break;
                } else {
                    buf.push(u8b[0]);
                }
            }
            if !buf.is_empty() {
                let ret = String::from_utf8(buf).unwrap();
                return ret.parse().unwrap();
            }
        }
    }
    pub fn read_vec<T: std::str::FromStr>(n: usize) -> Vec<T>
    where
        <T as FromStr>::Err: Debug,
    {
        (0..n).into_iter().map(|_| read::<T>()).collect::<Vec<T>>()
    }
    pub fn read_vec_sub1(n: usize) -> Vec<usize> {
        (0..n)
            .into_iter()
            .map(|_| read::<usize>() - 1)
            .collect::<Vec<usize>>()
    }
    pub fn read_mat<T: std::str::FromStr>(h: usize, w: usize) -> Vec<Vec<T>>
    where
        <T as FromStr>::Err: Debug,
    {
        (0..h)
            .into_iter()
            .map(|_| read_vec::<T>(w))
            .collect::<Vec<Vec<T>>>()
    }
}
use procon_reader::*;
/*************************************************************************************
*************************************************************************************/

fn main() {
    Solver::new().solve();
}

#[derive(Clone, Copy)]
struct Param {
    eff: i64,
    key_power: usize,
    key_exca_th: usize,
    observe_power: usize,
    observe_exca_th: usize,
    connect_power: usize,
    connect_exca_th: usize,
    evalw: usize,
    fix_rate: usize,
    delta_range_inv: i64,
    delta_cost_w: usize,
    atk_eval_rate: usize,
}
static mut PARAM: Param = Param {
    eff: 14,
    key_power: 68,
    key_exca_th: 43,
    observe_power: 68,
    observe_exca_th: 43,
    connect_power: 68,
    connect_exca_th: 43,
    evalw: 9,
    fix_rate: 241,
    delta_range_inv: 18,
    delta_cost_w: 1,
    atk_eval_rate: 0
};
const PARAMS: [Param; 8] = [
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
    Param {eff: 14, key_power: 68, key_exca_th: 43, observe_power: 68, observe_exca_th: 43, connect_power: 68, connect_exca_th: 43, evalw: 9, fix_rate: 241, delta_range_inv: 18, delta_cost_w: 1, atk_eval_rate: 8},
];

fn get_param() -> &'static Param {
    unsafe { &PARAM }
}
fn set_param(param: Param) {
    unsafe {
        PARAM = param;
    }
}
const INF: i64 = 1i64 << 60;
const DIR4: [(i64, i64); 4] = [(0, 1), (1, 0), (-1, 0), (0, -1)];

mod state {
    const HMAX: usize = 5000;
    use crate::get_param;
    use crate::procon_reader::*;
    use crate::union_find::UnionFind;
    use crate::usize_move_delta::MoveDelta;
    pub struct State {
        n: usize,
        fixed: Vec<Vec<bool>>,
        cum_attack: Vec<Vec<usize>>,
        evaluate: Vec<Vec<Option<usize>>>,
        brokens: UnionFind,
        brokens_or_water: UnionFind,
        brokens_or_near_water: UnionFind,
        c: usize,
    }
    impl State {
        pub fn new(n: usize, c: usize) -> Self {
            Self {
                n,
                fixed: vec![vec![false; n]; n],
                cum_attack: vec![vec![0; n]; n],
                evaluate: vec![vec![None; n]; n],
                brokens: UnionFind::new(n * n),
                brokens_or_water: UnionFind::new(n * n + 1),
                brokens_or_near_water: UnionFind::new(n * n + 1),
                c,
            }
        }
        pub fn set_near_water(&mut self, y: usize, x: usize) {
            let n = self.n;
            self.brokens_or_near_water.unite(n * n, y * n + x)
        }
        pub fn set_water(&mut self, y: usize, x: usize) {
            let n = self.n;
            self.brokens_or_water.unite(n * n, y * n + x)
        }
        pub fn is_watered(&mut self, y: usize, x: usize) -> bool {
            let n = self.n;
            self.brokens_or_water.same(n * n, y * n + x)
        }
        pub fn is_near_watered(&mut self, y: usize, x: usize) -> bool {
            let n = self.n;
            self.brokens_or_near_water.same(n * n, y * n + x)
        }
        pub fn is_spacially_connected(&mut self, y0: usize, x0: usize, y1: usize, x1: usize) -> bool {
            let n = self.n;
            self.brokens.same(y0 * n + x0, y1 * n + x1)
        }
        pub fn n(&self) -> usize {
            self.n
        }
        pub fn delta_line(&self, y: usize, x: usize, ny: usize, nx: usize, given_power: usize) -> i64 {
            let y0 = std::cmp::min(y, ny);
            let x0 = std::cmp::min(x, nx);
            let y1 = std::cmp::max(y, ny);
            let x1 = std::cmp::max(x, nx);
            let mut valid_sm = 0;
            let mut valid_norm = 0;
            let mut empty_norm = 0;
            let mut delta = 0;
            let fix_rate = get_param().fix_rate;
            let eff = get_param().eff;
            let delta_range_inv = get_param().delta_range_inv;
            let n = self.n;
            let cost = |atk: usize| -> usize {
                let cnt = (atk + given_power - 1) / given_power;
                let costed_atk = cnt * (given_power + self.c);
                let w = get_param().delta_cost_w;
                (atk * w + costed_atk * (w - 1)) / w
            };
            for &(cy, cx) in [(y, x), (ny, nx)].iter() {
                // horizontal
                for &(dy_unit, dx_unit) in crate::DIR4.iter() {
                    for d in 0..(eff / delta_range_inv) {
                        let dy = dy_unit * d;
                        let dx = dx_unit * d;
                        if let Some(y) = cy.move_delta(dy, 0, n - 1) {
                            if let Some(x) = cx.move_delta(dx, 0, n - 1) {
                                if self.fixed[y][x] {
                                    valid_sm += cost(self.cum_attack[y][x]) * fix_rate;
                                    valid_norm += fix_rate;
                                } else if let Some(weight) = self.evaluate[y][x] {
                                    delta += cost(weight - self.cum_attack[y][x]);
                                    valid_sm += cost(weight);
                                    valid_norm += 1;
                                } else {
                                    empty_norm += 1;
                                }
                            }
                        }
                    }
                }
            }
            if y0 == y1 {
                // horizontal
                for x in x0..=x1 {
                    if self.fixed[y0][x] {
                        valid_sm += cost(self.cum_attack[y0][x]) * fix_rate;
                        valid_norm += fix_rate;
                    } else if let Some(weight) = self.evaluate[y0][x] {
                        delta += cost(weight - self.cum_attack[y0][x]);
                        valid_sm += cost(weight);
                        valid_norm += 1;
                    } else {
                        empty_norm += 1;
                    }
                }
            } else if x0 == x1 {
                // vertical
                for y in y0..=y1 {
                    if self.fixed[y][x0] {
                        valid_sm += cost(self.cum_attack[y][x0]) * fix_rate;
                        valid_norm += fix_rate;
                    } else if let Some(weight) = self.evaluate[y][x0] {
                        delta += cost(weight - self.cum_attack[y][x0]);
                        valid_sm += cost(weight);
                        valid_norm += 1;
                    } else {
                        empty_norm += 1;
                    }
                }
            } else {
                unreachable!();
            }
            delta as i64 + (valid_sm * empty_norm) as i64 / valid_norm as i64
        }
        pub fn excavate_line(&mut self, y: usize, x: usize, ny: usize, nx: usize, force_all: bool, given_power: usize, given_exca_th: usize) {
            let n = self.n();
            if y == ny {
                // horizontal
                let dx = if x < nx { 1 } else { -1 };
                let mut px = x;
                loop {
                    if self.excavate_point(y, px, true, given_power, given_exca_th) && !force_all {
                        return;
                    }
                    if px == nx {
                        break;
                    }
                    if let Some(pnx) = px.move_delta(dx, 0, n - 1) {
                        px = pnx;
                    } else {
                        break;
                    }
                }
            } else if x == nx {
                // vertical
                let dy = if y < ny { 1 } else { -1 };
                let mut py = y;
                loop {
                    if self.excavate_point(py, x, true, given_power, given_exca_th) && !force_all {
                        return;
                    }
                    if py == ny {
                        break;
                    }
                    if let Some(pny) = py.move_delta(dy, 0, n - 1) {
                        py = pny;
                    } else {
                        break;
                    }
                }
            } else {
                unreachable!();
            }
        }
        pub fn is_line_full(&mut self, y: usize, x: usize, ny: usize, nx: usize) -> bool {
            let y0 = std::cmp::min(y, ny);
            let x0 = std::cmp::min(x, nx);
            let y1 = std::cmp::max(y, ny);
            let x1 = std::cmp::max(x, nx);
            if y0 == y1 {
                // horizontal
                for x in x0..=x1 {
                    if !self.fixed[y0][x] {
                        return false;
                    }
                }
            } else if x0 == x1 {
                // horizontal
                for y in y0..=y1 {
                    if !self.fixed[y][x0] {
                        return false;
                    }
                }
            } else {
                unreachable!();
            }
            true
        }
        pub fn excavate_point(&mut self, y: usize, x: usize, force_break: bool, given_power: usize, given_exca_th: usize) -> bool {
            //            let power = get_param().power;
            let power = if let Some(eval) = self.evaluate[y][x] {
                let eval_power = std::cmp::max(eval, given_power);
                let w = get_param().atk_eval_rate;
                (eval_power * (w - 1) + given_power) / w
            } else {
                given_power
            };
            if self.fixed[y][x] {
                return false;
            }
            loop {
                self.cum_attack[y][x] += power;
                if Self::attack(y, x, power) {
                    self.fixed[y][x] = true;
                    self.evaluate[y][x] = Some(self.cum_attack[y][x]);
                    break;
                }
                if !force_break && (self.cum_attack[y][x] >= given_exca_th) {
                    let evalw = get_param().evalw;
                    self.evaluate[y][x] =
                        Some((HMAX + self.cum_attack[y][x] * (evalw - 1)) / evalw);
                    break;
                }
            }
            if self.fixed[y][x] {
                let n = self.fixed.len();
                for &(dy, dx) in crate::DIR4.iter() {
                    if let Some(ny) = y.move_delta(dy, 0, n - 1) {
                        if let Some(nx) = x.move_delta(dx, 0, n - 1) {
                            if self.fixed[ny][nx] {
                                self.brokens.unite(y * n + x, ny * n + nx);
                                self.brokens_or_water.unite(y * n + x, ny * n + nx);
                                self.brokens_or_near_water.unite(y * n + x, ny * n + nx);
                            }
                        }
                    }
                }
            }
            self.fixed[y][x]
        }
        fn attack(y: usize, x: usize, p: usize) -> bool {
            //return true;
            println!("{} {} {}", y, x, p);
            match read::<i64>() {
                0 => false,
                1 => true,
                2 => {
                    std::process::exit(0);
                }
                _ => {
                    std::process::exit(1);
                }
            }
        }
        pub fn is_fixed(&self, y: usize, x: usize) -> bool {
            self.fixed[y][x]
        }
    }
}
use state::State;

struct Solver {
    n: usize,
    waters: Vec<(usize, usize)>,
    houses: Vec<(usize, usize)>,
    state: State,
}

impl Solver {
    fn new() -> Self {
        let n = read::<usize>();
        let w = read::<usize>();
        let k = read::<usize>();
        let c = read::<usize>();

        for cs in 0..8 {
            if (1 << cs) == c {
                set_param(PARAMS[cs]);
            }
        }
        let mut args = std::env::args().collect::<Vec<String>>();
        args.remove(0);
        if args.len() >= 12 {
            set_param( Param {
                eff: args[0].parse().unwrap(),
                key_power: args[1].parse().unwrap(),
                key_exca_th: args[2].parse().unwrap(),
                observe_power: args[3].parse().unwrap(),
                observe_exca_th: args[4].parse().unwrap(),
                connect_power: args[5].parse().unwrap(),
                connect_exca_th: args[6].parse().unwrap(),
                evalw: args[7].parse().unwrap(),
                fix_rate: args[8].parse().unwrap(),
                delta_range_inv: args[9].parse().unwrap(),
                delta_cost_w: args[10].parse().unwrap(),
                atk_eval_rate: args[11].parse().unwrap(),
            });
        }

        let mut waters = vec![];
        for _ in 0..w {
            waters.push((read::<usize>(), read::<usize>()));
        }
        let mut houses = vec![];
        for _ in 0..k {
            houses.push((read::<usize>(), read::<usize>()));
        }
        let state = State::new(n, c);
        Self {
            n,
            waters,
            houses,
            state,
        }
    }
    fn excavate_keypoints(state: &mut State, keypoints: &[(usize, usize)]) {
        for &(y, x) in keypoints.iter() {
            state.excavate_point(y, x, true, get_param().key_power, get_param().key_exca_th);
        }
    }
    fn excavate_observers(state: &mut State, observers: &[Vec<(usize, usize)>]) {
        for row in observers.iter() {
            for &(y, x) in row.iter() {
                state.excavate_point(y, x, false, get_param().observe_power, get_param().observe_exca_th);
            }
        }
    }
    fn calc_observers(n: usize) -> Vec<Vec<(usize, usize)>> {
        let mut observers = vec![];
        let eff = get_param().eff as usize;
        for y in ((eff / 2)..n).step_by(eff) {
            let mut row = vec![];
            for x in ((eff / 2)..n).step_by(eff) {
                row.push((y, x));
            }
            observers.push(row);
        }
        observers
    }
    fn calc_near_observers_near_keypoints(
        state: &mut State,
        keypoints: &[(usize, usize)],
        observers: &[Vec<(usize, usize)>],
    ) -> Vec<Vec<(usize, usize)>> {
        let mut near_observers_for_earh_keypoint = vec![vec![]; keypoints.len()];

        let m = observers.len();
        for (ki, &(ky, kx)) in keypoints.iter().enumerate() {
            for yi in 0..m {
                for xi in 0..m {
                    let (oy, ox) = observers[yi][xi];
                    if (oy as i64 - ky as i64).abs() >= get_param().eff {
                        continue;
                    }
                    if (ox as i64 - kx as i64).abs() >= get_param().eff {
                        continue;
                    }
                    near_observers_for_earh_keypoint[ki].push((yi, xi));
                }
            }
        }
        near_observers_for_earh_keypoint
    }
    fn connect_observers(
        &mut self,
        observers: &[Vec<(usize, usize)>],
        near_observers_for_earh_house: &[Vec<(usize, usize)>],
    ) {
        loop {
            let mut min_cost = None;
            let mut min_cost_pre = HashMap::new();
            let mut min_cost_watered_y = 0;
            let mut min_cost_watered_x = 0;
            let mut min_cost_hi = 0;
            let mut proc_any = false;
            'houseloop: for (hi, &(hy, hx)) in self.houses.iter().enumerate() {
                for &(near_oyi, near_oxi) in near_observers_for_earh_house[hi].iter() {
                    let (oy, ox) = observers[near_oyi][near_oxi];
                    if self.state.is_near_watered(oy, ox) {
                        continue 'houseloop;
                    }
                }
                proc_any = true;
                for &(near_oyi, near_oxi) in near_observers_for_earh_house[hi].iter() {
                    let mut que = BinaryHeap::new();
                    let mut dist = HashMap::new();
                    let mut pre = HashMap::new();
                    que.push(Reverse((0, near_oyi, near_oxi)));
                    dist.insert((near_oyi, near_oxi), 0);
                    let m = observers.len();
                    let mut near_watered_y = 0;
                    let mut near_watered_x = 0;
                    let mut near_watered_dist = None;
                    while let Some(Reverse((d, yi, xi))) = que.pop() {
                        if dist[&(yi, xi)] != d {
                            continue;
                        }
                        let y = observers[yi][xi].0;
                        let x = observers[yi][xi].1;

                        if self.state.is_near_watered(y, x) && near_watered_dist.chmin(d) {
                            near_watered_y = y;
                            near_watered_x = x;
                        }

                        for &(dy, dx) in crate::DIR4.iter() {
                            if let Some(nyi) = yi.move_delta(dy, 0, m - 1) {
                                if let Some(nxi) = xi.move_delta(dx, 0, m - 1) {
                                    let ny = observers[nyi][nxi].0;
                                    let nx = observers[nyi][nxi].1;
                                    let nd = d + self.state.delta_line(y, x, ny, nx, get_param().connect_power);
                                    if dist.entry((nyi, nxi)).or_insert(INF).chmin(nd) {
                                        pre.insert((ny, nx), (y, x));
                                        que.push(Reverse((nd, nyi, nxi)));
                                    }
                                }
                            }
                        }
                    }

                    if min_cost.chmin(near_watered_dist.unwrap()) {
                        min_cost_pre = pre;
                        min_cost_watered_y = near_watered_y;
                        min_cost_watered_x = near_watered_x;
                        min_cost_hi = hi;
                    }
                }
            }

            let mut to_y = min_cost_watered_y;
            let mut to_x = min_cost_watered_x;
            while let Some(&(from_y, from_x)) = min_cost_pre.get(&(to_y, to_x)) {
                self.state.excavate_line(to_y, to_x, from_y, from_x, false, get_param().connect_power, get_param().connect_exca_th);
                to_y = from_y;
                to_x = from_x;
            }

            if !proc_any {
                break;
            }
        }
    }
    fn set_near_water(state: &mut State, near_observers_for_each_water: &[Vec<(usize, usize)>], observers: &[Vec<(usize, usize)>]) {
        for vc in near_observers_for_each_water.iter() {
            for &(yi, xi) in vc.iter() {
                let (oy, ox) = observers[yi][xi];
                state.set_near_water(oy, ox);
            }
        }
    }
    fn connect_waters_to_near_observer(
        state: &mut State,
        observers: &[Vec<(usize, usize)>],
        near_observers_for_each_keypoints: &[Vec<(usize, usize)>],
        waters: &[(usize, usize)],
        houses: &[(usize, usize)],
    )
    {
        loop {
            for (wi, around) in near_observers_for_each_keypoints.iter().enumerate() {
                let (wy, wx) = waters[wi];
                let mut min_dist = None;
                let mut min_dist_t = None;
                for &(oyi, oxi) in around {
                    let (oy, ox) = observers[oyi][oxi];
                    let mut can_ignore = true;
                    for &(hy, hx) in houses.iter() {
                        if state.is_spacially_connected(hy, hx, oy, ox) {
                            if !state.is_watered(hy, hx) {
                                can_ignore = false;
                            } else {
                            }
                        }
                    }
                    if can_ignore {
                        continue;
                    }

                    for &(ty, tx) in [(oy, wx), (wy, ox)].iter(){
                        let d = state.delta_line(wy, wx, ty, tx, get_param().key_power) + state.delta_line(ty, tx, oy, ox, get_param().key_power);
                        if min_dist.chmin(d) {
                            min_dist_t = Some(((ty, tx), (oy, ox)));
                        }
                    }
                }
                if let Some(((ty, tx), (oy, ox))) = min_dist_t {
                    state.excavate_line(wy, wx, ty, tx, false, get_param().key_power, get_param().key_exca_th);
                    state.excavate_line(ty, tx, oy, ox, false, get_param().key_power, get_param().key_exca_th);
                }
            }
        }
    }
    fn connect_houses_to_near_observer(
        state: &mut State,
        observers: &[Vec<(usize, usize)>],
        near_observers_for_each_keypoints: &[Vec<(usize, usize)>],
        houses: &[(usize, usize)],
    )
    {
        for (hi, around) in near_observers_for_each_keypoints.iter().enumerate() {
            let (hy, hx) = houses[hi];
            loop {
                let mut min_dist = None;
                let mut min_dist_t = None;
                for &(oyi, oxi) in around {
                    let (oy, ox) = observers[oyi][oxi];
                    if !state.is_near_watered(oy, ox) {
                        continue;
                    }
                    if state.is_spacially_connected(hy, hx, oy, ox) {
                        continue;
                    }
                    for &(ty, tx) in [(oy, hx), (hy, ox)].iter(){
                        let d = state.delta_line(hy, hx, ty, tx, get_param().key_power) + state.delta_line(ty, tx, oy, ox, get_param().key_power);
                        if min_dist.chmin(d) {
                            min_dist_t = Some(((ty, tx), (oy, ox)));
                        }
                    }
                }
                if let Some(((ty, tx), (oy, ox))) = min_dist_t {
                    state.excavate_line(hy, hx, ty, tx, false, get_param().key_power, get_param().key_exca_th);
                    state.excavate_line(ty, tx, oy, ox, false, get_param().key_power, get_param().key_exca_th);
                } else {
                    break;
                }
            }
        }
    }
    fn solve(&mut self) {
        let observers = Self::calc_observers(self.n);
        Self::excavate_observers(&mut self.state, &observers);
        Self::excavate_keypoints(&mut self.state, &self.waters);
        Self::excavate_keypoints(&mut self.state, &self.houses);

        let near_observers_for_each_water = 
            Self::calc_near_observers_near_keypoints(&mut self.state, &self.waters, &observers);
        Self::set_near_water(&mut self.state, &near_observers_for_each_water, &observers);
        let near_observers_for_each_house =
            Self::calc_near_observers_near_keypoints(&mut self.state, &self.houses, &observers);
        self.connect_observers(&observers, &near_observers_for_each_house);
        Self::connect_houses_to_near_observer(&mut self.state, &observers, &near_observers_for_each_house, &self.houses);
        for &(wy, wx) in &self.waters {
            self.state.set_water(wy, wx);
        }
        Self::connect_waters_to_near_observer(&mut self.state, &observers, &near_observers_for_each_water, &self.waters, &self.houses);
    }
}
