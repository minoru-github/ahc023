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

#[cfg(feature = "visualize")]
use vis::*;

use ahc::common::*;
use my_lib::*;

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

    //ahc::solution_01::solve(&input);
    ahc::solution_02::solve(&input);

    let end_time = my_lib::time::update();
    let duration = end_time - start_time;
    eprintln!("{:?} ", duration);
}
