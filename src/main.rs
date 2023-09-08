#![allow(unused, non_snake_case, unused_macros)]
use itertools::Itertools;
use my_lib::*;
use proconio::{input, marker::*};
use rand::prelude::*;
use rand_distr::{Bernoulli, Normal, Uniform};
use rand_pcg::Mcg128Xsl64;
use std::{
    cell::RefCell,
    clone,
    cmp::{max, min},
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
    iter::FromIterator,
    mem::swap,
    ops::*,
    rc::Rc,
    slice::SliceIndex,
};

fn main() {
    let start_time = my_lib::time::update();

    let input = if std::env::args().len() >= 2 {
        let in_file = std::env::args().nth(1).unwrap();
        let input = std::fs::read_to_string(&in_file).unwrap_or_else(|_| {
            eprintln!("no such file: {}", in_file);
            std::process::exit(1)
        });
        parse_input(&input)
    } else {
        read_input()
    };

    // let (score, err) = match out {
    //     Ok(out) => compute_score(&input, &out),
    //     Err(err) => (0, err),
    // };
    // println!("Score = {}", score);
    // if err.len() > 0 {
    //     println!("{}", err);
    // }

    Sim::new(input.clone()).run();

    let end_time = my_lib::time::update();
    let duration = end_time - start_time;
    eprintln!("{:?} ", duration);
}

#[derive(Debug, Clone)]
struct Land {
    space_area_mat: Vec<Vec<(usize, usize)>>, // 各点が属する領域の面積(h方向, w方向)
                                              // space: Vec<Vec<usize>>, // 各点が属する領域の番号
}

impl Land {
    fn new(H: usize, W: usize) -> Self {
        let space_area_mat = vec![vec![(0, 0); W]; H];

        Land { space_area_mat }
    }

    fn compute_area(&mut self, input: &Input) {
        // w方向
        for h in 0..input.H {
            let mut cnt = 1;
            for w in 0..=input.W - 2 {
                self.space_area_mat[h][w].1 = cnt;
                if !input.is_water_tate[h][w] {
                    cnt += 1;
                } else {
                    cnt = 1;
                }
            }
            self.space_area_mat[h][input.W - 1].1 = cnt;

            // 逆方向に探索し、最も大きいで更新する
            for w in (0..=input.W - 2).rev() {
                if !input.is_water_tate[h][w] {
                    self.space_area_mat[h][w].1 = self.space_area_mat[h][w + 1].1;
                }
            }
        }

        // h方向
        for w in 0..input.W {
            let mut cnt = 1;
            for h in 0..=input.H - 2 {
                self.space_area_mat[h][w].0 = cnt;
                if !input.is_water_yoko[h][w] {
                    cnt += 1;
                } else {
                    cnt = 1;
                }
            }
            self.space_area_mat[input.H - 1][w].0 = cnt;

            // 逆方向に探索し、最も大きいで更新する
            for h in (0..=input.H - 2).rev() {
                if !input.is_water_yoko[h][w] {
                    self.space_area_mat[h][w].0 = self.space_area_mat[h + 1][w].0;
                }
            }
        }

        //self.debug();
    }

    fn debug(&self) {
        for h in 0..self.space_area_mat.len() {
            for w in 0..self.space_area_mat[0].len() {
                print!(
                    "({}, {}) ",
                    self.space_area_mat[h][w].0, self.space_area_mat[h][w].1
                );
            }
            println!();
        }
    }
}

#[derive(Debug, Clone)]
pub struct State {
    score: usize,
}

impl State {
    fn new() -> Self {
        State { score: 0 }
    }

    fn change(&mut self, output: &mut Output, rng: &mut Mcg128Xsl64) {
        //let val = rng.gen_range(-3, 4);
        //self.x += val;
    }

    fn compute_score(&mut self) {
        //self.score = 0;
    }
}

#[derive(Debug, Clone)]
pub struct Sim {
    input: Input,
}

impl Sim {
    fn new(input: Input) -> Self {
        // TODO: impl input
        //dbg!(input.clone());
        Sim { input }
    }

    pub fn run(&mut self) {
        let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);
        let mut cnt = 0 as usize; // 試行回数

        let mut land = Land::new(self.input.H, self.input.W);
        land.compute_area(&self.input);

        //let mut initial_state = State::new();

        // eprintln!("{} ", cnt);
        // eprintln!("{} ", best_state.score);
    }
}

mod solver {
    use super::*;

    pub fn mountain(
        best_state: &mut State,
        state: &State,
        best_output: &mut Output,
        output: &Output,
    ) {
        //! bese_state(self)を更新する。

        // 最小化の場合は > , 最大化の場合は < 。
        if best_state.score > state.score {
            *best_state = state.clone();
            *best_output = output.clone();
        }
    }

    const T0: f64 = 2e3;
    //const T1: f64 = 6e2; // 終端温度が高いと最後まで悪いスコアを許容する
    const T1: f64 = 6e1; // 終端温度が高いと最後まで悪いスコアを許容する
    pub fn simulated_annealing(
        best_state: &mut State,
        state: &State,
        best_output: &mut Output,
        output: &Output,
        current_time: f64,
        rng: &mut Mcg128Xsl64,
    ) {
        //! 焼きなまし法
        //! https://scrapbox.io/minyorupgc/%E7%84%BC%E3%81%8D%E3%81%AA%E3%81%BE%E3%81%97%E6%B3%95

        static mut T: f64 = T0;
        static mut CNT: usize = 0;
        let temperature = unsafe {
            CNT += 1;
            if CNT % 100 == 0 {
                let t = current_time / my_lib::time::LIMIT;
                T = T0.powf(1.0 - t) * T1.powf(t);
            }
            T
        };

        // 最大化の場合
        let delta = (best_state.score as f64) - (state.score as f64);
        // 最小化の場合
        //let delta = (state.score as f64) - (best_state.score as f64);

        let prob = f64::exp(-delta / temperature).min(1.0);

        if delta < 0.0 {
            *best_state = state.clone();
            *best_output = output.clone();
        } else if rng.gen_bool(prob) {
            *best_state = state.clone();
            *best_output = output.clone();
        }
    }
}

mod my_lib {
    //! 基本的に問題によらず変えない自作ライブラリ群
    use super::*;
    pub mod time {
        //! 時間管理モジュール
        pub fn update() -> f64 {
            static mut STARTING_TIME_MS: Option<f64> = None;
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let time_ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
            unsafe {
                let now = match STARTING_TIME_MS {
                    Some(starting_time_ms) => time_ms - starting_time_ms,
                    None => {
                        STARTING_TIME_MS = Some(time_ms);
                        0.0 as f64
                    }
                };
                now
            }
        }

        // TODO: set LIMIT
        pub const LIMIT: f64 = 0.3;
    }

    pub trait Mat<S, T> {
        fn set(&mut self, p: S, value: T);
        fn get(&self, p: S) -> T;
        fn swap(&mut self, p1: S, p2: S);
    }

    impl<T> Mat<&Point, T> for Vec<Vec<T>>
    where
        T: Copy,
    {
        fn set(&mut self, p: &Point, value: T) {
            self[p.y][p.x] = value;
        }

        fn get(&self, p: &Point) -> T {
            self[p.y][p.x]
        }

        fn swap(&mut self, p1: &Point, p2: &Point) {
            let tmp = self[p1.y][p1.x];
            self[p1.y][p1.x] = self[p2.y][p2.x];
            self[p2.y][p2.x] = tmp;
        }
    }

    impl<T> Mat<Point, T> for Vec<Vec<T>>
    where
        T: Copy,
    {
        fn set(&mut self, p: Point, value: T) {
            self[p.y][p.x] = value;
        }

        fn get(&self, p: Point) -> T {
            self[p.y][p.x]
        }

        fn swap(&mut self, p1: Point, p2: Point) {
            let tmp = self[p1.y][p1.x];
            self[p1.y][p1.x] = self[p2.y][p2.x];
            self[p2.y][p2.x] = tmp;
        }
    }

    impl Add for Point {
        type Output = Result<Point, &'static str>;
        fn add(self, rhs: Self) -> Self::Output {
            let (x, y) = if cfg!(debug_assertions) {
                // debugではオーバーフローでpanic発生するため、オーバーフローの溢れを明確に無視する(※1.60場合。それ以外は不明)
                (self.x.wrapping_add(rhs.x), self.y.wrapping_add(rhs.y))
            } else {
                (self.x + rhs.x, self.y + rhs.y)
            };

            unsafe {
                if let Some(width) = WIDTH {
                    if x >= width || y >= width {
                        return Err("out of range");
                    }
                }
            }

            Ok(Point { x, y })
        }
    }

    static mut WIDTH: Option<usize> = None;

    #[derive(Debug, Clone, PartialEq, Eq, Copy)]
    pub struct Point {
        pub x: usize, // →
        pub y: usize, // ↑
    }

    impl Point {
        pub fn new(x: usize, y: usize) -> Self {
            Point { x, y }
        }

        pub fn set_width(width: usize) {
            unsafe {
                WIDTH = Some(width);
            }
        }
    }

    pub trait SortFloat {
        fn sort(&mut self);
        fn sort_rev(&mut self);
    }

    impl SortFloat for Vec<f64> {
        fn sort(&mut self) {
            //! 浮動小数点としてNANが含まれないことを約束されている場合のsort処理<br>
            //! 小さい順
            self.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        fn sort_rev(&mut self) {
            //! 浮動小数点としてNANが含まれないことを約束されている場合のsort処理<br>  
            //! 大きい順
            self.sort_by(|a, b| b.partial_cmp(a).unwrap());
        }
    }

    pub trait EvenOdd {
        fn is_even(&self) -> bool;
        fn is_odd(&self) -> bool;
    }

    impl EvenOdd for usize {
        fn is_even(&self) -> bool {
            self % 2 == 0
        }

        fn is_odd(&self) -> bool {
            self % 2 != 0
        }
    }
}

mod procon_input {
    use std::{any::type_name, io::*};

    fn read_block<T: std::str::FromStr>() -> T {
        let mut s = String::new();
        let mut buf = [0];
        loop {
            stdin().read(&mut buf).expect("can't read.");
            let c = buf[0] as char;
            if c == ' ' {
                break;
            }
            // for Linux
            if c == '\n' {
                break;
            }
            // for Windows
            if c == '\r' {
                // pop LR(line feed)
                stdin().read(&mut buf).expect("can't read.");
                break;
            }
            s.push(c);
        }
        s.parse::<T>()
            .unwrap_or_else(|_| panic!("can't parse '{}' to {}", s, type_name::<T>()))
    }

    pub fn read_i() -> i64 {
        read_block::<i64>()
    }

    pub fn read_ii() -> (i64, i64) {
        (read_block::<i64>(), read_block::<i64>())
    }

    pub fn read_iii() -> (i64, i64, i64) {
        (
            read_block::<i64>(),
            read_block::<i64>(),
            read_block::<i64>(),
        )
    }

    pub fn read_iiii() -> (i64, i64, i64, i64) {
        (
            read_block::<i64>(),
            read_block::<i64>(),
            read_block::<i64>(),
            read_block::<i64>(),
        )
    }

    pub fn read_u() -> usize {
        read_block::<usize>()
    }

    pub fn read_uu() -> (usize, usize) {
        (read_block::<usize>(), read_block::<usize>())
    }

    pub fn read_uuu() -> (usize, usize, usize) {
        (
            read_block::<usize>(),
            read_block::<usize>(),
            read_block::<usize>(),
        )
    }

    pub fn read_uuuu() -> (usize, usize, usize, usize) {
        (
            read_block::<usize>(),
            read_block::<usize>(),
            read_block::<usize>(),
            read_block::<usize>(),
        )
    }

    pub fn read_f() -> f64 {
        read_block::<f64>()
    }

    pub fn read_ff() -> (f64, f64) {
        (read_block::<f64>(), read_block::<f64>())
    }

    pub fn read_c() -> char {
        read_block::<char>()
    }

    pub fn read_cc() -> (char, char) {
        (read_block::<char>(), read_block::<char>())
    }

    fn read_line() -> String {
        let mut s = String::new();
        stdin().read_line(&mut s).expect("can't read.");
        s.trim()
            .parse()
            .unwrap_or_else(|_| panic!("can't trim in read_line()"))
    }

    pub fn read_vec<T: std::str::FromStr>() -> Vec<T> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<T>()))
            })
            .collect()
    }

    pub fn read_i_vec() -> Vec<i64> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<i64>()))
            })
            .collect()
    }

    pub fn read_u_vec() -> Vec<usize> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<usize>()))
            })
            .collect()
    }

    pub fn read_f_vec() -> Vec<f64> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<f64>()))
            })
            .collect()
    }

    pub fn read_c_vec() -> Vec<char> {
        read_line()
            .split_whitespace()
            .map(|e| {
                e.parse()
                    .unwrap_or_else(|_| panic!("can't parse '{}' to {}", e, type_name::<char>()))
            })
            .collect()
    }

    pub fn read_line_as_chars() -> Vec<char> {
        //! a b c d -> \[a, b, c, d]
        read_line().as_bytes().iter().map(|&b| b as char).collect()
    }

    pub fn read_string() -> String {
        //! abcd -> "abcd"
        read_block::<String>()
    }

    pub fn read_string_as_chars() -> Vec<char> {
        //! abcd -> \[a, b, c, d]
        read_block::<String>().chars().collect::<Vec<char>>()
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    pub T: usize,
    pub H: usize,
    pub W: usize,
    pub i0: usize,                     // 0-based
    pub is_water_yoko: Vec<Vec<bool>>, // 水路の有無　横 オリジナルではh
    pub is_water_tate: Vec<Vec<bool>>, // 水路の有無　縦 オリジナルではv
    pub K: usize,                      // 作物の種類数
    pub S: Vec<usize>,                 // 1-based  作物kはS[k]月までに植える
    pub D: Vec<usize>,                 // 1-based　作物kはD[k]月に収穫する
}

impl Input {
    pub fn is_valid_point(&self, x: usize, y: usize) -> bool {
        x < self.H && y < self.W
    }
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {} {}", self.T, self.H, self.W, self.i0)?;
        for h in &self.is_water_yoko {
            writeln!(
                f,
                "{}",
                h.iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect::<String>()
            )?;
        }
        for v in &self.is_water_tate {
            writeln!(
                f,
                "{}",
                v.iter()
                    .map(|&b| if b { '1' } else { '0' })
                    .collect::<String>()
            )?;
        }
        writeln!(f, "{}", self.K)?;
        for k in 0..self.K {
            writeln!(f, "{} {}", self.S[k], self.D[k])?;
        }
        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let mut f = proconio::source::once::OnceSource::from(f);
    input! {
        from &mut f,
        T: usize,
        H: usize,
        W: usize,
        i0: usize,
        h1: [Chars; H - 1],
        v1: [Chars; H],
        K: usize,
        SD: [(usize, usize); K],
    }
    let is_water_yoko = h1
        .iter()
        .map(|i| i.iter().map(|&b| b == '1').collect())
        .collect();
    let is_water_tate = v1
        .iter()
        .map(|i| i.iter().map(|&b| b == '1').collect())
        .collect();
    let S = SD.iter().map(|x| x.0).collect();
    let D = SD.iter().map(|x| x.1).collect();
    Input {
        T,
        H,
        W,
        i0,
        is_water_yoko,
        is_water_tate,
        K,
        S,
        D,
    }
}

pub fn read_input() -> Input {
    input! {
        T: usize,
        H: usize,
        W: usize,
        i0: usize,
        h1: [Chars; H - 1],
        v1: [Chars; H],
        K: usize,
        SD: [(usize, usize); K],
    }
    let is_water_yoko = h1
        .iter()
        .map(|i| i.iter().map(|&b| b == '1').collect())
        .collect();
    let is_water_tate = v1
        .iter()
        .map(|i| i.iter().map(|&b| b == '1').collect())
        .collect();
    let S = SD.iter().map(|x| x.0).collect();
    let D = SD.iter().map(|x| x.1).collect();
    Input {
        T,
        H,
        W,
        i0,
        is_water_yoko,
        is_water_tate,
        K,
        S,
        D,
    }
}

fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr>(
    token: Option<&str>,
    lb: T,
    ub: T,
) -> Result<T, String> {
    if let Some(v) = token {
        if let Ok(v) = v.parse::<T>() {
            if v < lb || ub < v {
                Err(format!("Out of range: {}", v))
            } else {
                Ok(v)
            }
        } else {
            Err(format!("Parse error: {}", v))
        }
    } else {
        Err("Unexpected EOF".to_owned())
    }
}

#[derive(Clone, Debug)]
pub struct Output {
    pub M: usize,
    pub works: Vec<Work>,
}

#[derive(Clone, Debug)]
pub struct Work {
    pub k: usize, // 0-based
    pub i: usize, // 0-based
    pub j: usize, // 0-based
    pub s: usize, // 1-based
}

impl Output {
    pub fn new() -> Self {
        Output {
            M: 0,
            works: Vec::new(),
        }
    }

    pub fn validate(&self, input: &Input) -> Result<(), String> {
        // check range
        for Work {
            k, i: _, j: _, s, ..
        } in self.works.iter().cloned()
        {
            let ub = input.S[k];
            if s > ub {
                return Err(format!("Cannot plant crop {} after month {}", k + 1, ub));
            }
        }

        // check duplicates
        {
            let mut items = BTreeSet::new();
            for Work { k, .. } in self.works.iter() {
                if !items.insert(*k) {
                    return Err(format!("Crop {} is planted more than once", k + 1));
                }
            }
        }
        Ok(())
    }
}

pub fn compute_score(input: &Input, out: &Output) -> (i64, String) {
    if let Err(msg) = out.validate(input) {
        return (0, msg);
    }

    let mut scheduled_works: Vec<Vec<Work>> = vec![vec![]; input.T + 1];
    for w in out.works.iter().cloned() {
        scheduled_works[w.s].push(w)
    }

    let mut workspace = vec![vec![None; input.W]; input.H];
    let adj = {
        let mut adj = vec![vec![Vec::new(); input.W]; input.H];
        for i in 0..input.H {
            for j in 0..input.W {
                if i + 1 < input.H && !input.is_water_yoko[i][j] {
                    adj[i + 1][j].push((i, j));
                    adj[i][j].push((i + 1, j));
                }
                if j + 1 < input.W && !input.is_water_tate[i][j] {
                    adj[i][j + 1].push((i, j));
                    adj[i][j].push((i, j + 1))
                }
            }
        }
        adj
    };

    let si = input.i0;
    let sj = 0;
    let start = (si, sj);
    let mut score = 0;

    for t in 1..=input.T {
        // beginning of month t
        {
            // check reachability
            if !scheduled_works[t].is_empty() {
                let mut visited = vec![vec![false; input.W]; input.H];

                if workspace[si][sj].is_none() {
                    let mut q = VecDeque::new();
                    q.push_back(start);
                    visited[si][sj] = true;

                    while !q.is_empty() {
                        let Some((x, y)) = q.pop_front() else { unreachable!() };
                        assert!(workspace[x][y].is_none());
                        for (x1, y1) in adj[x][y].iter().cloned() {
                            if input.is_valid_point(x1, y1)
                                && workspace[x1][y1].is_none()
                                && !visited[x1][y1]
                            {
                                q.push_back((x1, y1));
                                visited[x1][y1] = true;
                            }
                        }
                    }
                }

                for &Work { k, i, j, .. } in &scheduled_works[t] {
                    if !visited[i][j] {
                        return (
                            0,
                            format!(
                                "{} is scheduled at unreachable position {}, {}",
                                k + 1,
                                i,
                                j
                            ),
                        );
                    }
                }
            }

            // update workspace
            for &Work { k, i, j, s, .. } in &scheduled_works[t] {
                if let Some((k1, _)) = workspace[i][j] {
                    return (
                        0,
                        format!("Block ({}, {}) is occupied by crop {}", i, j, k1 + 1),
                    );
                } else {
                    workspace[i][j] = Some((k, s))
                }
            }
        }

        // end of month t; harvest crops
        let can_start = {
            if let Some((k, _s)) = workspace[si][sj] {
                input.D[k] == t
            } else {
                true
            }
        };

        if can_start {
            let mut q = VecDeque::new();
            q.push_back(start);
            let mut visited = vec![vec![false; input.W]; input.H];
            visited[si][sj] = true;

            while !q.is_empty() {
                let Some((i, j)) = q.pop_front() else { unreachable!() };
                if let Some((k, s)) = workspace[i][j] {
                    if input.D[k] == t {
                        workspace[i][j] = None;
                        let span = t - s + 1;
                        // this should hold because we do not
                        // allow planting crop k after month S[k]
                        assert!(span >= input.D[k] - input.S[k] + 1);
                        score += input.D[k] - input.S[k] + 1;
                    } else if input.D[k] < t {
                        return (
                            0,
                            format!("Cannot harvest crop {} in month {}", k + 1, input.D[k]),
                        );
                    }
                }

                for &(i1, j1) in &adj[i][j] {
                    assert!(input.is_valid_point(i1, j1));
                    let is_blocked = {
                        if let Some((k, _s)) = workspace[i1][j1] {
                            input.D[k] > t
                        } else {
                            false
                        }
                    };
                    if !is_blocked && !visited[i1][j1] {
                        q.push_back((i1, j1));
                        visited[i1][j1] = true;
                    }
                }
            }
        }
    }

    (
        ((score as u64 * 1_000_000) as f64 / (input.H * input.W * input.T) as f64).round() as i64,
        String::new(),
    )
}
