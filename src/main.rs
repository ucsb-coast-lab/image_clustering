#![allow(dead_code)]
#![allow(unused_parens)]
#![allow(unused_imports)]
use image::ImageBuffer;
use image::*;

mod algorithm_lib;
mod externals_lib;
mod visualization_lib;

use algorithm_lib::*;
use externals_lib::*;
use visualization_lib::scatter_image_sandbox;

use rand::{Rng, SeedableRng};
use std::env;

fn main() {
    let export_dir = "pictures/processed/";
    let args: Vec<_> = env::args_os().collect();
    // capture_images("/dev/video0",10,200,"pictures/captured");

    // For kmeans clustering
    let num_clusters = 3; // Controls the number of clusters
    let ratio = 0.0; // Controls the ratio of how important Euclidean pixel distance vs. color variation is

    // For density-based clustering
    // let epsilon: f64 = 4.;
    // epsilon = 7 for the kelp.jpg example
    // let min_nghbrs = 5;
    // let size = 100;

    for i in 1..args.len() {
        //println!("Attempting to open: {:?}",args[i]);
        let path = args[i].clone().into_string().unwrap().to_string();
        let parsed_path: Vec<_> = path.split("/").collect();
        //println!("parsed_path: {:?}", parsed_path);
        let clustered_img = kmeans_cluster_image(&path, num_clusters, ratio);
        // let clustered_img = db_cluster_image(&path, epsilon, min_nghbrs,size);
        clustered_img
            .save(export_dir.to_owned() + &parsed_path[parsed_path.len() - 1])
            .expect("Error while saving clusterd image");
    }
}
