use clap::Parser;
use std::fs;
use std::io::prelude::*;

use rust::extract_dataset;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to original meta dataset file
    #[arg(short, long)]
    src_file: std::path::PathBuf,
    /// The output file path for writing extracted subset1 dataset
    #[arg(long)]
    dst_train_file: std::path::PathBuf,
    /// The output file path for writing extracted subset2 dataset
    #[arg(long)]
    dst_test_file: std::path::PathBuf,
    /// ratio
    #[arg(long)]
    train_ratio: f64,
}

fn main() {
    let args = Args::parse();

    let categories = extract_dataset(args.src_file);

    let mut buffers = (
        fs::File::create(args.dst_train_file).unwrap(),
        fs::File::create(args.dst_test_file).unwrap(),
    );

    // subset1
    for (_category_name, instance_map) in categories.iter() {
        let num_inst = instance_map.len();
        let num_train = (num_inst as f64 * args.train_ratio).ceil() as usize;
        for (i, (_instance_id, instances)) in instance_map.iter().enumerate() {
            let buffer;
            if i < num_train {
                buffer = &mut buffers.0;
            } else {
                buffer = &mut buffers.1;
            }
            for line in instances {
                buffer.write_fmt(format_args!("{}\n", line)).unwrap();
            }
        }
    }
}
