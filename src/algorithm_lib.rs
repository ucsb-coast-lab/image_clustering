use crate::*;

#[derive(Debug, Clone, Copy)]
pub struct KMeansPixel {
    pub position: (u32, u32),
    pub rgb: (u8, u8, u8),
    pub cluster: usize,
}

impl KMeansPixel {
    fn print(&self) {
        print!(
            "Point: ({:.2},{:.2}), rgb: ({},{},{}), cluster: {}",
            self.position.0, self.position.1, self.rgb.0, self.rgb.1, self.rgb.2, self.cluster
        );
    }
}

pub fn get_distance(a: (u32, u32), b: (u32, u32)) -> f64 {
    let (a, b) = ((a.0 as f64, a.1 as f64), (b.0 as f64, b.0 as f64));
    let dist = ((a.0 - b.0).abs().powf(2.0) + (a.1 - b.1).abs().powf(2.0)).sqrt();
    dist
}

// A good way to explore this variation would be to work on developing a method
// for mapping the x,y,r,g,b differences into a 2D parameter space and graphing
// that, at which point the standard 2D kmeans algorithm can work

pub fn get_variation(a: KMeansPixel, b: KMeansPixel) -> f64 {
    let dist = ((a.position.0 as f64 - b.position.0 as f64).abs().powf(2.0)
        + (a.position.1 as f64 - b.position.1 as f64).abs().powf(2.0))
    .sqrt();
    let color_error = color_distance(a, b);

    dist + (color_error / 2.0)
}

pub fn convert_to_subspace(data: Vec<KMeansPixel>) -> Vec<(f64, f64)> {
    let mut subspace: Vec<(f64, f64)> = Vec::new();

    //let (mut dist_param, mut color_param): (f64,f64) = (0.0,0.0);
    for pixel in &data {
        let dist_param = 0.0;
        let color_param = 0.0;

        subspace.push((dist_param, color_param))
    }

    subspace
}

pub fn color_distance(a: KMeansPixel, b: KMeansPixel) -> f64 {
    // Euclidean distance
    /*
    let cd = ((a.rgb.0 as f64 - b.rgb.0 as f64).abs().powf(2.0)
        + (a.rgb.1 as f64 - b.rgb.1 as f64).abs().powf(2.0)
        + (a.rgb.2 as f64 - b.rgb.2 as f64).abs().powf(2.0))
    .powf(0.5);
    */

    //let cd = (a.rgb.2 as f64 - b.rgb.2 as f64).abs() + (a.rgb.1 as f64 - b.rgb.1 as f64).abs();
    //let cd = (a.rgb.0 as f64 - b.rgb.0 as f64).abs() / (a.rgb.2 as f64 - b.rgb.2 as f64).abs();

    // Wikipedia human perception of color:
    let cd = if a.rgb.0 < 128 {
        (2.0 * (a.rgb.0 as f64 - b.rgb.0 as f64).powf(2.0)
            + 4.0 * (a.rgb.1 as f64 - b.rgb.1 as f64).powf(2.0)
            + 3.0 * (a.rgb.2 as f64 - b.rgb.2 as f64).powf(2.0))
        .powf(0.5)
    } else {
        (3.0 * (a.rgb.0 as f64 - b.rgb.0 as f64).powf(2.0)
            + 4.0 * (a.rgb.1 as f64 - b.rgb.1 as f64).powf(2.0)
            + 2.0 * (a.rgb.2 as f64 - b.rgb.2 as f64).powf(2.0))
        .powf(0.5)
    };
    cd
}

// Converts an image to a vector of KMeanPixel structures, and returns that and the images dimensions in a tuple
// Pixels are written to a vector in rows (row 0 all written, then row 1 all written, row 2, etc.)
pub fn build_kmeans_pixel_list_from_image(
    img_path: &str,
    clusters: Vec<(u8, u8, u8)>,
) -> (Vec<KMeansPixel>, u32, u32) {
    let mut img = image::open(img_path).expect("Couldn't open the image");
    let (w, h) = img.dimensions();
    println!("Image dimensions: ({},{})", w, h);

    let max: f32 = 235.;
    let min: f32 = 10.;
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x,y);
            let (mut r, mut g, mut b) = (pixel[0] as f32,pixel[1] as f32,pixel[2] as f32);
            r = ((r / 255.0) * (max - min)) + min;
            g = ((g / 255.0) * (max - min)) + min;
            b = ((b / 255.0) * (max - min)) + min;
            img.put_pixel(x,y,image::Rgba([r as u8, g as u8, b as u8, 0]));
        }
    }
    img.save("normalized.png").expect("Couldn't write the normalized image");

    let mut kmeans_pixels: Vec<KMeansPixel> = Vec::with_capacity(w as usize * h as usize); // Pre-allocating this cluster meaningfully improves performance
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            let position = (x, y);

            let mut assigned_cluster = clusters[0];
            let mut km_pixel = KMeansPixel {
                position: position,
                rgb: (pixel[0], pixel[1], pixel[2]),
                cluster: 0,
            };

            // Assigns easy pixel it's proper cluster
            for i in 0..clusters.len() {
                //let cen_px = img.get_pixel(cluster.0 as u32, cluster.1 as u32);
                let temp_cluster_pixel: KMeansPixel = KMeansPixel {
                    position: (0, 0),
                    rgb: clusters[i],
                    cluster: i,
                };

                let km_pixel_cluster: KMeansPixel = KMeansPixel {
                    position: (0, 0),
                    rgb: assigned_cluster,
                    cluster: km_pixel.cluster,
                };

                // If the difference in color between the km_pixel and a cluster-based KMeansPixel is less than
                // the difference in color between the km_pixel and it's assigned cluster, the assigned_cluster
                // value is changed to match
                if color_distance(km_pixel.clone(), temp_cluster_pixel.clone())
                    < color_distance(km_pixel.clone(), km_pixel_cluster.clone())
                {
                    assigned_cluster = clusters[i].clone();
                }
            }
            let index = clusters
                .iter()
                .position(|&value| value == assigned_cluster)
                .expect("There was an error assigned a cluster index");
            km_pixel.cluster = index;
            //println!("km_cluster: {:?}",km_pixel);

            kmeans_pixels.push(km_pixel);
        }
    }
    (kmeans_pixels, w, h)
}

pub fn cluster_image(img_path: &str, num_clusters: usize) -> ImageBuffer<Rgb<u8>, Vec<u8>> {

    // Here, we're doing clustering in the RGB subspace alone
    let mut clusters: Vec<(u8, u8, u8)> = build_clusters(num_clusters);
    let (mut kmeans_pixels, w, h) = build_kmeans_pixel_list_from_image(img_path, clusters.clone());

    let mut iteration = 0;
    let mut prev_clusters: Vec<(u8, u8, u8)> = Vec::with_capacity(num_clusters);
    let mut cval = 255; // A value to determine when the variation between cluster iterations has converged (Convergence VALue)
    let max_iterations = 10;

    // MAIN KMEANS LOOP
    // As long as the cluster centroids haven't converged and we haven't cycled through a certain number of iterations,
    // keep updating the cluster values
    while (cval > num_clusters * 2) && (iteration < max_iterations) {
        println!("*** ITERATION #{}", iteration);

        for point in &mut kmeans_pixels {
            // Calculate the distance to each cluster in the list. If the distance is smaller than the one that exists, replace
            for i in 0..clusters.len() {
                let cluster = clusters[i];

                // This is pixel updated each time the cluster list is iterated through
                let temp_cluster_pixel: KMeansPixel = KMeansPixel {
                    position: (0, 0),
                    rgb: clusters[i],
                    cluster: i,
                };

                let assigned_cluster_pixel: KMeansPixel = KMeansPixel {
                    position: point.position,
                    rgb: clusters[point.cluster],
                    cluster: point.cluster,
                };

                if color_distance(point.clone(), temp_cluster_pixel.clone())
                    < color_distance(point.clone(), assigned_cluster_pixel.clone())
                {
                    let index = clusters
                        .iter()
                        .position(|&value| value == cluster.clone())
                        .unwrap();
                    point.cluster = index;
                    //println!("")
                }
            }
        }

        // Information for print debugging
        /*
        let mut count = 0;
        for cluster in &clusters {
            println!("cluster ({:.2},{:.2}): ", cluster.0, cluster.1,cluster.2);
            for point in &kmeans_pixels {
                if clusters[point.cluster] == cluster.clone() {
                    //point.print();
                    //println!(" #{}", count);
                    count += 1;
                }
            }
        }
        */

        // Update the location of the clusters
        prev_clusters = clusters.clone();
        for i in 0..clusters.len() {
            let mut r_sum: u32 = 0;
            let mut g_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut counter: u32 = 0;

            for point in &kmeans_pixels {
                //if kmeans_pixels[j].cluster == clusters[i] {
                //if color_distance(clusters[point.cluster].clone(), clusters[i].clone()) < 0.05 {
                if clusters[point.cluster].clone() == clusters[i].clone() {
                    //println!("Pixel {:?} assigned to cluster #{}",point.position,i);
                    counter = counter + 1;
                    r_sum = r_sum + point.rgb.0 as u32;
                    g_sum = g_sum + point.rgb.1 as u32;
                    b_sum = b_sum + point.rgb.2 as u32;
                    //println!("({},{},{})",)
                }
            }
            if counter == 0 {
                counter += 1;
            }
            clusters[i] = (
                (r_sum / counter) as u8,
                (g_sum / counter) as u8,
                (b_sum / counter) as u8,
            );
        }

        print!("\n");
        print!("Current clusters: ");
        for cluster in &clusters {
            print!("({},{},{}), ", cluster.0, cluster.1, cluster.2);
        }
        print!("\n");
        cval = check_cluster_sum_diff(clusters.clone(), prev_clusters);
        println!("The cval in iteration #{} was {}", iteration, cval);
        iteration += 1;

        // Let's check that our clusters are valid, and if not, we'll reset
        let mut counts: Vec<_> = Vec::new();
        for n in 0..clusters.len() {
            counts.push(0u32);
            for point in &kmeans_pixels {
                if point.cluster == n {
                    counts[n] += 1;
                }
            }
        }
        println!("Counts per cluster: {:?}", counts);
        // IMPORTANT: If the point in any given cluster drops to zero, we start over
        // and assign the cluster new starting points
        for points_in_cluster in &counts {
            if points_in_cluster.clone() == 0 {
                iteration = 0;
                let new_clusters = build_clusters(num_clusters);
                for i in 0..num_clusters {
                    clusters[i] = new_clusters[i];
                }
            }
        }

    }
    println!("{} underwent {} iterations", img_path, iteration);

    // BUILDING CLUSTERED IMAGE
    // Built clusters, now iterating through pixels to re-build clustered image
    let mut clustered_img = image::ImageBuffer::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let ppx = kmeans_pixels[(w * y + x) as usize];
            let r = clusters[ppx.cluster].0;
            let g = clusters[ppx.cluster].1;
            let b = clusters[ppx.cluster].2;
            clustered_img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }
    clustered_img

    // Sleeping the thread here can help us visualize the clustering process
    //std::thread::sleep(std::time::Duration::from_millis(1500));
}

// This function is used to check that the variation between cluster points and their centriod are decreasing
// appropriately, and can be used as a way to establish convergence rather than relying upon a set number of
// iterations
fn check_cluster_sum_diff(a: Vec<(u8, u8, u8)>, b: Vec<(u8, u8, u8)>) -> usize {
    let mut diff = 0;
    if a.len() != b.len() {
        panic!("We somehow lost a cluster somehow?");
    } else {
        let mut a_sum = 0;
        let mut b_sum = 0;
        for i in 0..a.len() {
            a_sum += (a[i].0 as isize + a[i].1 as isize + a[i].2 as isize);
            println!("a_sum = {}", a_sum);
            b_sum += (b[i].0 as isize + b[i].2 as isize + b[i].2 as isize);
            println!("b_sum = {}", b_sum);
        }
        diff = (a_sum - b_sum).abs();
        println!("diff: {}", diff);
    }
    diff as usize
}

fn build_clusters(num_clusters: usize) -> Vec<(u8,u8,u8)> {
    let mut cluster_rng = rand::thread_rng();
    let mut clusters: Vec<(u8,u8,u8)> = Vec::with_capacity(num_clusters);
    println!("Number of clusters is: {}",num_clusters);
    for i in 0..num_clusters {
        // Randomly generates initial clusters
        clusters.push((
            cluster_rng.gen_range(50,200) as u8,
            cluster_rng.gen_range(50,200) as u8,
            cluster_rng.gen_range(50,200) as u8,
        ));
        println!("Cluster {}: {:?}",i, clusters[i]);
    }
    clusters
}
