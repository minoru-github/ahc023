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
    path,
    process::exit,
    rc::Rc,
    slice::SliceIndex,
};

use crate::common::*;
use my_lib::*;

pub fn solve(input: &Input) {
    let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);
    let mut cnt = 0 as usize; // 試行回数

    let mut land = Land::new(input.i0, 0, input.H, input.W);
    land.compute_shotest_path(input);
    land.count_childs_path();

    #[cfg(feature = "visualize")]
    vis::Visualizer::new(
        input.H,
        input.W,
        input.i0,
        input.is_water_yoko.clone(),
        input.is_water_tate.clone(),
    )
    .add_land()
    //.add_shortest_path_dist(&land.dist)
    .add_cnt_childs(&land.cnt_childs)
    .output_svg();
}

#[derive(Debug, Clone)]
struct Land {
    sy: usize,                      // スタート地点のy座標
    sx: usize,                      // スタート地点のx座標
    H: usize,                       // マップの高さ
    W: usize,                       // マップの幅
    space_id: Vec<Vec<i32>>,        // 各点が属する領域の番号
    dist: Vec<Vec<i32>>,            // 各点までの最短距離
    from: Vec<Vec<(usize, usize)>>, // 各点への最短経路での一つ前の点
    cnt_childs: Vec<Vec<usize>>,    // 各点の子孫の数
    path_set: BTreeSet<usize>,
    entry_points: Vec<(usize, usize)>,
}

impl Land {
    fn new(sy: usize, sx: usize, H: usize, W: usize) -> Self {
        Land {
            sy,
            sx,
            H,
            W,
            space_id: vec![vec![-1; W]; H],
            dist: vec![vec![0; W]; H],
            from: vec![vec![(0, 0); W]; H],
            cnt_childs: vec![vec![0; W]; H],
            path_set: BTreeSet::new(),
            entry_points: vec![],
        }
    }

    fn compute_shotest_path(&mut self, input: &Input) {
        let mut que = VecDeque::new();
        que.push_back((self.sy, self.sx));

        let mut has_seen = vec![vec![false; self.W]; self.H];
        has_seen[self.sy][self.sx] = true;

        let mut dist = vec![vec![(self.W + self.H) as i32; self.W]; self.H];
        dist[self.sy][self.sx] = 0;

        let mut from = vec![vec![(0, 0); self.W]; self.H];
        from[self.sy][self.sx] = (self.sy, self.sx);

        while let Some((y, x)) = que.pop_front() {
            for (dy, dx) in &[(0, 1), (0, -1), (1, 0), (-1, 0)] {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;
                if ny < 0 || nx < 0 || ny >= self.H as i32 || nx >= self.W as i32 {
                    continue;
                }

                let ny = ny as usize;
                let nx = nx as usize;

                match (dy, dx) {
                    (0, 1) => {
                        if input.is_water_tate[y][x] {
                            continue;
                        }
                    }
                    (0, -1) => {
                        if input.is_water_tate[y][x - 1] {
                            continue;
                        }
                    }
                    (1, 0) => {
                        if input.is_water_yoko[y][x] {
                            continue;
                        }
                    }
                    (-1, 0) => {
                        if input.is_water_yoko[y - 1][x] {
                            continue;
                        }
                    }
                    _ => unreachable!(),
                }

                if has_seen[ny][nx] {
                    continue;
                }
                has_seen[ny][nx] = true;

                dist[ny][nx] = dist[y][x] + 1;
                from[ny][nx] = (y, x);
                que.push_back((ny, nx));
            }
        }

        self.dist = dist;
        self.from = from;
    }

    fn count_childs_path(&mut self) {
        let is_start = |y: usize, x: usize| (y, x) == (self.sy, self.sx);

        let mut cnt_childs = vec![vec![0; self.W]; self.H];
        for y in 0..self.H {
            for x in 0..self.W {
                if is_start(y, x) {
                    continue;
                }

                let mut ny = y;
                let mut nx = x;
                while !is_start(ny, nx) {
                    let (py, px) = self.from[ny][nx];
                    cnt_childs[py][px] += 1;
                    ny = py;
                    nx = px;
                }
            }
        }

        self.cnt_childs = cnt_childs;
    }

    fn decide_path(&mut self) {
        let mut path_set = BTreeSet::new();

        let path_th = 5;
        for y in 0..self.H {
            for x in 0..self.W {
                if self.cnt_childs[y][x] >= 5 {
                    let index = y * self.W + x;
                    path_set.insert(index);
                }
            }
        }

        self.path_set = path_set;
    }

    fn search_entry_points(&mut self) {
        let mut entry_points = vec![];

        for y in 0..self.H {
            for x in 0..self.W {
                if self.cnt_childs[y][x] == 0 {
                    entry_points.push((y, x));
                }
            }
        }

        self.entry_points = entry_points;
    }
}
