#![allow(unused, non_snake_case, unused_macros)]
use itertools::Itertools;
use num_integer::Integer;
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
    process::exit,
    rc::Rc,
    slice::SliceIndex,
};

pub mod my_lib {
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
    pub y: usize, // 0-based
    pub x: usize, // 0-based
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
            k, y: _, x: _, s, ..
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
        for y in 0..input.H {
            for x in 0..input.W {
                if y + 1 < input.H && !input.is_water_yoko[y][x] {
                    adj[y + 1][x].push((y, x));
                    adj[y][x].push((y + 1, x));
                }
                if x + 1 < input.W && !input.is_water_tate[y][x] {
                    adj[y][x + 1].push((y, x));
                    adj[y][x].push((y, x + 1))
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

                for &Work { k, y, x, .. } in &scheduled_works[t] {
                    if !visited[y][x] {
                        return (
                            0,
                            format!(
                                "{} is scheduled at unreachable position {}, {}",
                                k + 1,
                                y,
                                x
                            ),
                        );
                    }
                }
            }

            // update workspace
            for &Work { k, y, x, s, .. } in &scheduled_works[t] {
                if let Some((k1, _)) = workspace[y][x] {
                    return (
                        0,
                        format!("Block ({}, {}) is occupied by crop {}", y, x, k1 + 1),
                    );
                } else {
                    workspace[y][x] = Some((k, s))
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
                let Some((y, x)) = q.pop_front() else { unreachable!() };
                if let Some((k, s)) = workspace[y][x] {
                    if input.D[k] == t {
                        workspace[y][x] = None;
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

                for &(y1, x1) in &adj[y][x] {
                    assert!(input.is_valid_point(y1, x1));
                    let is_blocked = {
                        if let Some((k, _s)) = workspace[y1][x1] {
                            input.D[k] > t
                        } else {
                            false
                        }
                    };
                    if !is_blocked && !visited[y1][x1] {
                        q.push_back((y1, x1));
                        visited[y1][x1] = true;
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
