#![allow(dead_code)]
use image::ImageBuffer;
use image::*;

mod algorithm_lib;
mod visualization_lib;

use algorithm_lib::*;
use visualization_lib::*;

use rand::Rng;
use std::ffi::OsString;
// use std::error::Error;
use std::env;

fn main() {
    /*let img_path =  match get_first_arg() {
        Ok(path) => path,
        Err(error) => panic!(
            "Error: Provided file_path string was not considered valid because {}",
            error
        ),
    };

    let path = match img_path.into_string() {
        Ok(path) => path,
        Err(_error) => panic!("Couldn't convert OsString to String"),
    };*/

    let export_dir = "pictures/processed/";
    let args: Vec<_> = env::args_os().collect();
    let num_clusters = 3;
    let mut clusters: Vec<(u8, u8, u8)> = Vec::with_capacity(num_clusters);
    for i in 1..args.len() {
        //println!("Attempting to open: {:?}",args[i]);
        let path = args[i].clone().into_string().unwrap().to_string();
        let parsed_path: Vec<_> = path.split("/").collect();
        println!("parsed_path: {:?}", parsed_path);
        let (clustered_img, clusters) = cluster_image(&path, num_clusters);
        clustered_img
            .save(export_dir.to_owned() + &parsed_path[parsed_path.len() - 1])
            .expect("Error while saving clusterd image");
    }
}
