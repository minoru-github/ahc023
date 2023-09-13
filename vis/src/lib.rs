#![allow(unused, non_snake_case, unused_macros)]
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
use svg::node::element::path::Data as SvgData;
use svg::node::element::Path as SvgPath;
use svg::node::element::Rectangle as SvgRect;
use svg::node::element::Text as SvgText;
use svg::node::Text as TextContent;
use svg::Document;

const CHART_MARGIN: f32 = 25.0;
const BOX_SIZE: f32 = 50.0;

pub struct Visualizer {
    pub H: usize,
    pub W: usize,
    pub start_y: usize,                // 0-based
    pub is_water_yoko: Vec<Vec<bool>>, // 水路の有無　横 オリジナルではh
    pub is_water_tate: Vec<Vec<bool>>, // 水路の有無　縦 オリジナルではv
    svg: Document,
}

impl Visualizer {
    pub fn new(
        H: usize,
        W: usize,
        start_y: usize,                // 0-based
        is_water_yoko: Vec<Vec<bool>>, // 水路の有無　横 オリジナルではh
        is_water_tate: Vec<Vec<bool>>, // 水路の有無　縦 オリジナルではv
    ) -> Self {
        let svg = Document::new()
            .set(
                "viewBox",
                (
                    0,
                    0,
                    (CHART_MARGIN * 2.0 + W as f32 * BOX_SIZE) as i32,
                    (CHART_MARGIN * 2.0 + H as f32 * BOX_SIZE) as i32,
                ),
            )
            .set("width", (CHART_MARGIN * 2.0 + BOX_SIZE * W as f32) as i64)
            .set("height", (CHART_MARGIN * 2.0 + BOX_SIZE * H as f32) as i64);

        Self {
            H,
            W,
            start_y,
            is_water_yoko,
            is_water_tate,
            svg,
        }
    }

    pub fn output_svg(&mut self) {
        svg::save("image.svg", &self.svg).unwrap();
    }

    pub fn add_land(&mut self) -> &mut Self {
        self.add_base();
        self.add_wall();
        self.add_water();

        self
    }

    fn add_base(&mut self) {
        let color = "rgba(255, 192, 203, 0.5)";

        let mut svg = self.svg.clone();

        for y in 0..=self.H {
            let mut data = SvgData::new();
            let start = (CHART_MARGIN, CHART_MARGIN + y as f32 * BOX_SIZE);
            let end = (self.W as f32 * BOX_SIZE, 0);
            data = data.move_to(start).line_by(end).close();

            let path = self.create_line_instance(data, color, 1);

            svg = svg.add(path);
        }

        for x in 0..=self.W {
            let mut data = SvgData::new();
            let start = (CHART_MARGIN + x as f32 * BOX_SIZE, CHART_MARGIN);
            let end = (0, self.H as f32 * BOX_SIZE);
            data = data.move_to(start).line_by(end).close();

            let path = self.create_line_instance(data, color, 1);

            svg = svg.add(path);
        }

        self.svg = svg;
    }

    fn add_wall(&mut self) {
        let color = "black";

        let mut svg = self.svg.clone();

        let mut data = SvgData::new();
        // 上の壁
        {
            let start = (CHART_MARGIN, CHART_MARGIN);
            let end = (self.W as f32 * BOX_SIZE, 0);
            data = data.move_to(start).line_by(end);
        }
        // 下の壁
        {
            let start = (CHART_MARGIN, CHART_MARGIN + self.H as f32 * BOX_SIZE);
            let end = (self.W as f32 * BOX_SIZE, 0);
            data = data.move_to(start).line_by(end);
        }
        // 右の壁
        {
            let start = (CHART_MARGIN + self.W as f32 * BOX_SIZE, CHART_MARGIN);
            let end = (0, self.H as f32 * BOX_SIZE);
            data = data.move_to(start).line_by(end);
        }
        // 左の壁(start_yの位置は除く)
        {
            let start = (CHART_MARGIN, CHART_MARGIN);
            let end = (0, self.start_y as f32 * BOX_SIZE);
            data = data.move_to(start).line_by(end);

            let start = (
                CHART_MARGIN,
                CHART_MARGIN + (self.start_y + 1) as f32 * BOX_SIZE,
            );
            let end = (0, (self.H - self.start_y - 1) as f32 * BOX_SIZE);
            data = data.move_to(start).line_by(end);
        }

        let path = self.create_line_instance(data, color, 4);
        svg = svg.add(path);

        self.svg = svg;
    }

    fn add_water(&mut self) {
        let color = "black";

        let mut svg = self.svg.clone();

        // 縦向きの水路
        for y in 0..self.H {
            for x in 0..self.W - 1 {
                if self.is_water_tate[y][x] {
                    let mut data = SvgData::new();
                    let start = (
                        CHART_MARGIN + (x + 1) as f32 * BOX_SIZE,
                        CHART_MARGIN + y as f32 * BOX_SIZE,
                    );
                    let end = (0, BOX_SIZE);
                    data = data.move_to(start).line_by(end).close();

                    let path = self.create_line_instance(data, color, 4);

                    svg = svg.add(path);
                }
            }
        }

        // 横向きの水路
        for y in 0..self.H - 1 {
            for x in 0..self.W {
                if self.is_water_yoko[y][x] {
                    let mut data = SvgData::new();
                    let start = (
                        CHART_MARGIN + x as f32 * BOX_SIZE,
                        CHART_MARGIN + (y + 1) as f32 * BOX_SIZE,
                    );
                    let end = (BOX_SIZE, 0);
                    data = data.move_to(start).line_by(end).close();

                    let path = self.create_line_instance(data, color, 4);

                    svg = svg.add(path);
                }
            }
        }

        self.svg = svg;
    }

    pub fn add_space_id(
        &mut self,
        space_id: &Vec<Vec<i32>>, // 各点が属する領域の番号
    ) -> &mut Self {
        let color_table = self.create_id_color_table();

        let mut svg = self.svg.clone();

        for y in 0..self.H {
            for x in 0..self.W {
                if space_id[y][x] == -1 {
                    let mut data = SvgData::new();
                    let start = (
                        CHART_MARGIN + x as f32 * BOX_SIZE,
                        CHART_MARGIN + y as f32 * BOX_SIZE,
                    );
                    let rect = SvgRect::new()
                        .set("x", start.0)
                        .set("y", start.1)
                        .set("width", BOX_SIZE)
                        .set("height", BOX_SIZE)
                        .set("fill", "rgba(255, 0, 0, 0.5)");
                    svg = svg.add(rect);
                } else {
                    let mut data = SvgData::new();
                    let start = (
                        CHART_MARGIN + x as f32 * BOX_SIZE,
                        CHART_MARGIN + y as f32 * BOX_SIZE,
                    );
                    let color = &color_table[space_id[y][x] as usize] as &str;
                    let rect = SvgRect::new()
                        .set("x", start.0)
                        .set("y", start.1)
                        .set("width", BOX_SIZE)
                        .set("height", BOX_SIZE)
                        .set("fill", color);
                    svg = svg.add(rect);
                }

                let mut text = SvgText::new()
                    .set("x", CHART_MARGIN + x as f32 * BOX_SIZE + BOX_SIZE / 2.0)
                    .set("y", CHART_MARGIN + y as f32 * BOX_SIZE + BOX_SIZE / 2.0)
                    .set("text-anchor", "middle")
                    .set("dominant-baseline", "central")
                    .set("font-size", 20)
                    .set("fill", "black");

                let content = TextContent::new(space_id[y][x].to_string());
                text = text.add(content);

                svg = svg.add(text);
            }
        }

        self.svg = svg;

        self
    }

    fn create_id_color_table(&self) -> Vec<String> {
        let mut color_table = Vec::new();

        let mut rng: Mcg128Xsl64 = rand_pcg::Pcg64Mcg::new(890482);

        let mut ids = BTreeSet::new();
        for y in 0..self.H {
            for x in 0..self.W {
                ids.insert(y * self.W + x);
            }
        }

        for id in ids {
            let r: u8 = rng.gen_range(0, 240);
            let g: u8 = rng.gen_range(20, 255);
            let b: u8 = rng.gen_range(20, 255);

            // (r,g,b)を文字列の"rgb(r,g,b,a)"に変換
            let mut color = String::from("rgba(");
            color.push_str(&r.to_string());
            color.push_str(",");
            color.push_str(&g.to_string());
            color.push_str(",");
            color.push_str(&b.to_string());
            color.push_str(", 0.25)");

            color_table.push(color);
        }

        color_table
    }

    pub fn add_shortest_path_dist(&mut self, dist: &Vec<Vec<i32>>) -> &mut Self {
        let mut svg = self.svg.clone();

        let dist_to_color_string = |dist: i32| {
            let max = (self.H + self.W) as i32;
            let cnt = min(max, dist);
            let r = (cnt as f32 / max as f32 * 255.0) as u8;
            let g = 0;
            let b = 0;
            let a = 0.5;
            format!("rgba({}, {}, {}, {})", r, g, b, a)
        };

        for y in 0..self.H {
            for x in 0..self.W {
                let mut data = SvgData::new();
                let start = (
                    CHART_MARGIN + x as f32 * BOX_SIZE,
                    CHART_MARGIN + y as f32 * BOX_SIZE,
                );
                let color = dist_to_color_string(dist[y][x]);
                let rect = SvgRect::new()
                    .set("x", start.0)
                    .set("y", start.1)
                    .set("width", BOX_SIZE)
                    .set("height", BOX_SIZE)
                    .set("fill", color);
                svg = svg.add(rect);

                let mut text = SvgText::new()
                    .set("x", CHART_MARGIN + x as f32 * BOX_SIZE + BOX_SIZE / 2.0)
                    .set("y", CHART_MARGIN + y as f32 * BOX_SIZE + BOX_SIZE / 2.0)
                    .set("text-anchor", "middle")
                    .set("dominant-baseline", "central")
                    .set("font-size", 20)
                    .set("fill", "black");

                let content = TextContent::new(dist[y][x].to_string());

                text = text.add(content);

                svg = svg.add(text);
            }
        }

        self.svg = svg;

        self
    }

    pub fn add_cnt_childs(&mut self, cnt_childs: &Vec<Vec<usize>>) -> &mut Self {
        let mut svg = self.svg.clone();

        let cnt_to_color_string = |cnt: usize| {
            let max = 10;
            let cnt = min(max, cnt);
            let r = (cnt as f32 / max as f32 * 255.0) as u8;
            let g = 0;
            let b = 0;
            let a = 0.5;
            format!("rgba({}, {}, {}, {})", r, g, b, a)
        };

        for y in 0..self.H {
            for x in 0..self.W {
                let mut data = SvgData::new();
                let start = (
                    CHART_MARGIN + x as f32 * BOX_SIZE,
                    CHART_MARGIN + y as f32 * BOX_SIZE,
                );
                let color = cnt_to_color_string(cnt_childs[y][x]);
                let rect = SvgRect::new()
                    .set("x", start.0)
                    .set("y", start.1)
                    .set("width", BOX_SIZE)
                    .set("height", BOX_SIZE)
                    .set("fill", color);
                svg = svg.add(rect);

                let mut text = SvgText::new()
                    .set("x", CHART_MARGIN + x as f32 * BOX_SIZE + BOX_SIZE / 2.0)
                    .set("y", CHART_MARGIN + y as f32 * BOX_SIZE + BOX_SIZE / 2.0)
                    .set("text-anchor", "middle")
                    .set("dominant-baseline", "central")
                    .set("font-size", 20)
                    .set("fill", "black");

                let content = TextContent::new(cnt_childs[y][x].to_string());
                text = text.add(content);

                svg = svg.add(text);
            }
        }

        self.svg = svg;

        self
    }

    fn create_line_instance(&self, data: SvgData, color: &str, width: usize) -> SvgPath {
        SvgPath::new()
            .set("fill", "none")
            .set("stroke", color)
            .set("stroke-width", width)
            .set("d", data)
    }
}
