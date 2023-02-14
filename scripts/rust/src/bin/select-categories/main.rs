use clap::Parser;
use rust::extract_dataset;
use std::fs;
use std::io::prelude::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The path to original meta dataset file
    #[arg(short, long)]
    src_file: std::path::PathBuf,
    /// The output file path for writing extracted meta dataset file
    #[arg(short, long)]
    dst_file: std::path::PathBuf,
    /// category list
    #[arg(short, long)]
    categories: Vec<String>,
    /// verbose
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    let categories = extract_dataset(args.src_file);

    let mut buffer = fs::File::create(args.dst_file).unwrap();

    for (category_name, instance_map) in categories.iter() {
        if !args.categories.contains(category_name) {
            continue;
        }

        for (_instance_id, instances) in instance_map.iter() {
            for line in instances {
                buffer.write_fmt(format_args!("{}\n", line)).unwrap();
            }
        }
    }
}
