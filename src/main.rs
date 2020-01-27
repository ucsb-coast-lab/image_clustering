#![allow(dead_code)]
#![allow(unused_parens)]
#![allow(unused_imports)]
use image::ImageBuffer;
use image::*;

mod algorithm_lib;
mod visualization_lib;
mod externals_lib;

use algorithm_lib::*;
use visualization_lib::scatter_image_sandbox;
use externals_lib::*;

use rand::{Rng,SeedableRng};
use std::env;

fn main() {
    let export_dir = "pictures/processed/";
    let args: Vec<_> = env::args_os().collect();
    // capture_images("/dev/video0",10,200,"pictures/captured");
    let num_clusters = 4;
    for i in 1..args.len() {
        //println!("Attempting to open: {:?}",args[i]);
        let path = args[i].clone().into_string().unwrap().to_string();
        let parsed_path: Vec<_> = path.split("/").collect();
        //println!("parsed_path: {:?}", parsed_path);
        let clustered_img = cluster_image(&path, num_clusters);
        clustered_img
            .save(export_dir.to_owned() + &parsed_path[parsed_path.len() - 1])
            .expect("Error while saving clusterd image");
    }
    //scatter_image_sandbox("pictures/submerged_kelp.png","pictures/submerged_kelp_scatter.png");
}
