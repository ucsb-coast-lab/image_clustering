#![allow(dead_code)]

use plotlib::page::Page;
use plotlib::repr::Scatter;
use plotlib::style::{PointMarker, PointStyle};
use plotlib::view::ContinuousView;

use image::GenericImageView;

use rand::Rng;
use std::process::Command;

use crate::algorithm_lib::*;

pub fn kmeans_plot(kmeans_list: Vec<CPixel>, clusters: Vec<(f64, f64)>, path_root: String) {
    // Checks the total number of elemnts being visualized
    let mut vis_vec: Vec<Vec<(f64, f64)>> = Vec::new();
    for _ in &clusters {
        vis_vec.push(Vec::new());
    }

    for point in &kmeans_list {
        let mut counter = 0;
        for cluster in &clusters {
            if get_distance(
                (
                    clusters[point.assigned_cluster].0 as u32,
                    clusters[point.assigned_cluster].1 as u32,
                ),
                (cluster.clone().0 as u32, cluster.clone().1 as u32),
            ) < 0.05
            {
                vis_vec[counter].push((point.x as f64, point.y as f64));
            }
            counter += 1;
        }
    }

    let mut element_sum = 0;
    for j in 0..vis_vec.len() {
        element_sum = element_sum + vis_vec[j].len();
    }
    /*println!(
        "The total number of elements in vis_vec is: {}",
        element_sum
    );*/

    let mut scatter_plots: Vec<Scatter> = Vec::new();
    for i in 0..clusters.len() {
        let mut rng = rand::thread_rng();
        let color = format!("#{}", rng.gen_range(0, 999999).to_string(),);
        let s: Scatter = Scatter::from_slice(&vis_vec[i]).style(
            PointStyle::new()
                .marker(PointMarker::Square) // setting the marker to be a square
                .colour(&color),
        );
        let c: Scatter = Scatter {
            data: vec![clusters[i].clone()],
            style: PointStyle::new(),
        }
        .style(PointStyle::new().colour(color));
        scatter_plots.push(s);
        scatter_plots.push(c);
    }

    //let mut data3: Vec<(f64,f64)> = vec![(-1.6, -2.7),(2.0,1.0)];

    let mut v = ContinuousView::new()
        .x_range(-5., 5.)
        .y_range(-5., 5.)
        .x_label("x-axis")
        .y_label("y-axis");

    for i in 0..scatter_plots.len() {
        v.representations.push(Box::new(scatter_plots[i].clone()));
    }
    //v.add(scatter_plots[0]);

    // A page with a single view is then saved to an SVG file
    let svg_path = path_root.clone() + ".svg";
    let png_path = path_root.clone() + ".png";

    Page::single(&v).save(&svg_path).unwrap();
    Command::new("cairosvg")
        .arg(svg_path)
        .arg("-o")
        .arg(png_path)
        .output()
        .expect("failed to convert .svg file to .png file");

    // Sleeps the thread for visualization
    //std::thread::sleep(std::time::Duration::from_millis(1500));
    //println!("\n");
}

pub fn scatter_image_sandbox(img_path: &str, export_path: &str) {
    // let img_path = "kelp.jpg";
    // let export_path = "clustered_img.png";
    let mut img = image::open(img_path).expect("Couldn't open the image");
    let (w, h) = img.dimensions();
    let num_clusters = 3;
    let mut clusters: Vec<Cluster> = Vec::with_capacity(num_clusters);
    let mut cluster_rng = rand::thread_rng();
    for i in 0..num_clusters {
        // Randomly generates initial clusters
        clusters.push(Cluster {
            x: cluster_rng.gen_range(0, w),
            y: cluster_rng.gen_range(0, h),
            r: cluster_rng.gen_range(50, 200) as u8,
            g: cluster_rng.gen_range(50, 200) as u8,
            b: cluster_rng.gen_range(50, 200) as u8,
        });
    }
    let (pixel_vec, w, h) = build_kmeans_pixel_list_from_image(img, clusters);
    let mut vis_vec: Vec<(f64, f64)> = Vec::with_capacity(pixel_vec.len());
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (0f64, 0f64, 0f64, 0f64);
    for pixel in &pixel_vec {
        vis_vec.push((pixel.rgb.1 as f64, pixel.rgb.2 as f64));
    }
    //println!("{:?}",vis_vec);

    let mut v = ContinuousView::new()
        .x_range(0., 255.)
        .y_range(0., 255.)
        .x_label("g")
        .y_label("b");
    let s: Scatter = Scatter::from_slice(&vis_vec).style(
        PointStyle::new()
            .marker(PointMarker::Square) // setting the marker to be a square
            .colour("123456"),
    );
    v.representations.push(Box::new(s));
    let svg_path = "cluster_img.svg";
    Page::single(&v).save(&svg_path).unwrap();
    Command::new("cairosvg")
        .arg(svg_path)
        .arg("-o")
        .arg(export_path)
        .output()
        .expect("failed to convert .svg file to .png file");
}
