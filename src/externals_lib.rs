use rscam::*;
use std::fs;
use std::io::Write;

pub fn capture_images(camera_address: &str, image_num: u64, interval: u64, export_dir: &str ) {
    let mut camera = rscam::new(camera_address).unwrap();

    camera.start(&rscam::Config {
        interval: (1, 30),      // 30 fps.
        resolution: (1280, 720),
        format: b"MJPG",
        ..Default::default()
    }).unwrap();

    for i in 0..(image_num as usize) {
        let frame = camera.capture().unwrap();
        let mut file = fs::File::create(&format!("{}/frame_{}.jpg",export_dir, i)).unwrap();
        file.write_all(&frame[..]).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(interval));
    }
}
