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

use crate::common::*;
use my_lib::*;

pub fn solve(input: &Input) {
    let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);
    let mut cnt = 0 as usize; // 試行回数

    let mut land = Land::new(input.H, input.W);
    land.compute_area(input);
    //land.debug_space_area();
    land.decide_space_id(input);
    land.decide_path(input);
    land.debug_space_id();

    #[cfg(feature = "visualize")]
    vis::Visualizer::new(
        input.H,
        input.W,
        input.i0,
        input.is_water_yoko.clone(),
        input.is_water_tate.clone(),
    )
    .add_land()
    .add_space_id(&land.space_id)
    .output_svg();
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpaceArea {
    height: usize,
    width: usize,
    sy: usize,
    sx: usize,
}

impl SpaceArea {
    fn new() -> Self {
        SpaceArea {
            height: 0,
            width: 0,
            sy: 0,
            sx: 0,
        }
    }
}

#[derive(Debug, Clone)]
struct Land {
    space_area_mat: Vec<Vec<SpaceArea>>, // 各点が属する領域の面積(h方向, w方向)
    space_id: Vec<Vec<i32>>,             // 各点が属する領域の番号
}

impl Land {
    fn new(H: usize, W: usize) -> Self {
        let space_area_mat = vec![vec![SpaceArea::new(); W]; H];
        let space_id = vec![vec![-1; W]; H];

        Land {
            space_area_mat,
            space_id,
        }
    }

    fn max_length(&self, y: usize, x: usize) -> usize {
        max(
            self.space_area_mat[y][x].height,
            self.space_area_mat[y][x].width,
        )
    }

    fn compute_area(&mut self, input: &Input) {
        // w方向
        for y in 0..input.H {
            let mut cnt = 1;

            // もし出入口が横にあるなら強制的にpathにするためにcnt高めにする
            if input.i0 == y {
                cnt = input.H * input.W;
            }

            let mut sx = 0;
            for x in 0..=input.W - 2 {
                self.space_area_mat[y][x].width = cnt;
                self.space_area_mat[y][x].sx = sx;

                if !input.is_water_tate[y][x] {
                    cnt += 1;
                } else {
                    cnt = 1;
                    sx = x + 1;
                }
                cnt = min(cnt, input.H * input.W);
            }
            self.space_area_mat[y][input.W - 1].width = cnt;
            self.space_area_mat[y][input.W - 1].sx = sx;

            // 逆方向に探索し、最も大きいで更新する
            for x in (0..=input.W - 2).rev() {
                if !input.is_water_tate[y][x] {
                    self.space_area_mat[y][x].width = self.space_area_mat[y][x + 1].width;
                }
            }
        }

        // h方向
        for w in 0..input.W {
            let mut cnt = 1;
            let mut sy = 0;
            for h in 0..=input.H - 2 {
                // もし出入口が横にあるなら強制的にpathにするためにcnt高めにする
                if input.i0 == h && w == 0 {
                    cnt = input.H * input.W;
                }

                self.space_area_mat[h][w].height = cnt;
                self.space_area_mat[h][w].sy = sy;
                if !input.is_water_yoko[h][w] {
                    cnt += 1;
                } else {
                    cnt = 1;
                    sy = h + 1;
                }
                cnt = min(cnt, input.H * input.W);
            }
            self.space_area_mat[input.H - 1][w].height = cnt;
            self.space_area_mat[input.H - 1][w].sy = sy;

            // 逆方向に探索し、最も大きいで更新する
            for h in (0..=input.H - 2).rev() {
                if !input.is_water_yoko[h][w] {
                    self.space_area_mat[h][w].height = self.space_area_mat[h + 1][w].height;
                }
            }
        }
    }

    fn debug_space_area(&self) {
        println!("★space_area_mat★");
        for h in 0..self.space_area_mat.len() {
            for w in 0..self.space_area_mat[0].len() {
                print!(
                    "({}, {});",
                    self.space_area_mat[h][w].height, self.space_area_mat[h][w].width
                );
            }
            println!();
        }
    }

    fn decide_space_id(&mut self, input: &Input) {
        let mut space_id = 0;
        for y in 0..input.H {
            for x in 0..input.W {
                if self.space_id[y][x] == -1 {
                    self.group_same_area((y, x), space_id, input);
                    space_id += 1;
                }
            }
        }
    }

    fn group_same_area(&mut self, (y, x): (usize, usize), id: i32, input: &Input) {
        let mut q = VecDeque::new();
        q.push_back((y, x));
        self.space_id[y][x] = id;

        while !q.is_empty() {
            let (h, w) = q.pop_front().unwrap();
            for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)].iter() {
                let (y1, x1) = (h as i32 + dy, w as i32 + dx);
                if y1 < 0 || y1 >= input.H as i32 || x1 < 0 || x1 >= input.W as i32 {
                    continue;
                }
                let (y1, x1) = (y1 as usize, x1 as usize);
                if self.space_id[y1][x1] != -1 {
                    continue;
                }

                if self.space_area_mat[h][w] == self.space_area_mat[y1][x1] {
                    match (dy, dx) {
                        (0, 1) => {
                            if !input.is_water_tate[h][w] {
                                self.space_id[y1][x1] = self.space_id[h][w];
                                q.push_back((y1, x1));
                            }
                        }
                        (0, -1) => {
                            if !input.is_water_tate[y1][x1] {
                                self.space_id[y1][x1] = self.space_id[h][w];
                                q.push_back((y1, x1));
                            }
                        }
                        (1, 0) => {
                            if !input.is_water_yoko[h][w] {
                                self.space_id[y1][x1] = self.space_id[h][w];
                                q.push_back((y1, x1));
                            }
                        }
                        (-1, 0) => {
                            if !input.is_water_yoko[y1][x1] {
                                self.space_id[y1][x1] = self.space_id[h][w];
                                q.push_back((y1, x1));
                            }
                        }
                        _ => unreachable!(),
                    }
                } else {
                    let max_len = self.max_length(h, w);
                    let max_len1 = self.max_length(y1, x1);

                    if max_len == max_len1 && max_len == input.H * input.W {
                        self.space_id[y1][x1] = self.space_id[h][w];
                        q.push_back((y1, x1));
                    }
                }
            }
        }
    }

    fn debug_space_id(&self) {
        //println!("★space_id★");
        for y in 0..self.space_id.len() {
            for x in 0..self.space_id[0].len() {
                print!("{};", self.space_id[y][x]);
            }
            println!();
        }
    }

    fn decide_path(&mut self, input: &Input) {
        let path_id = -1; //(input.H * input.W) as i32;
        for y in 0..input.H {
            for x in 0..input.W {
                for (dy, dx) in [(0, 1), (0, -1), (1, 0), (-1, 0)].iter() {
                    let (y1, x1) = (y as i32 + dy, x as i32 + dx);
                    if y1 < 0 || y1 >= input.H as i32 || x1 < 0 || x1 >= input.W as i32 {
                        continue;
                    }
                    let (y1, x1) = (y1 as usize, x1 as usize);

                    match (dy, dx) {
                        (0, 1) => {
                            if !input.is_water_tate[y][x] {
                                if (self.space_area_mat[y][x].height
                                    > self.space_area_mat[y1][x1].height)
                                    || (self.space_area_mat[y][x].width
                                        > self.space_area_mat[y1][x1].width)
                                {
                                    self.space_id[y][x] = path_id;
                                }
                            }
                        }
                        (0, -1) => {
                            if !input.is_water_tate[y1][x1] {
                                if (self.space_area_mat[y][x].height
                                    > self.space_area_mat[y1][x1].height)
                                    || (self.space_area_mat[y][x].width
                                        > self.space_area_mat[y1][x1].width)
                                {
                                    self.space_id[y][x] = path_id;
                                }
                            }
                        }
                        (1, 0) => {
                            if !input.is_water_yoko[y][x] {
                                if (self.space_area_mat[y][x].height
                                    > self.space_area_mat[y1][x1].height)
                                    || (self.space_area_mat[y][x].width
                                        > self.space_area_mat[y1][x1].width)
                                {
                                    self.space_id[y][x] = path_id;
                                }
                            }
                        }
                        (-1, 0) => {
                            if !input.is_water_yoko[y1][x1] {
                                if (self.space_area_mat[y][x].height
                                    > self.space_area_mat[y1][x1].height)
                                    || (self.space_area_mat[y][x].width
                                        > self.space_area_mat[y1][x1].width)
                                {
                                    self.space_id[y][x] = path_id;
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
    }
}
